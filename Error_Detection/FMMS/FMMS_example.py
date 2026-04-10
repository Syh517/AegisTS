from gen_meta_features import generate_meta_features
import numpy as np 
from joblib import load
import os 
from datasets import load_dataset
import time 
import sys
import pandas as pd 
from FMMS import FMMS
import torch
import config
import utils



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python FMMS_example.py <csv_path>")
        data_name = "SMD_machine-1-2"

        # sys.exit(1)
    else:
        data_name = sys.argv[1]  # e.g., "MSL_C-1"

    start_time = time.time()


        # Load test CSV
    try:
        calit2 = load_dataset("/home/yyy/TSC/TSClean/AutoClean/Error_Detection/Train_Datasets", data_dir=data_name)
        # Convert to pandas
        df = calit2["test"].to_pandas()
    except:
        ## Alternatively, if the above does not work, using exact path
        url = "/home/yyy/TSC/TSClean/AutoClean/Error_Detection/Train_Datasets/SMD_machine-1-2_test.csv"
        # url = "/home/yyy/TSC/TSClean/AutoClean/Error_Detection/Train_Datasets/{}/{}_test.csv".format(data_name.split('_')[0], data_name)
        df = pd.read_csv(url)

    X = df.iloc[:, 1:-1]  # remove timestamp and label

    trained_model_location = 'trained_models'
    meta_scalar = load(os.path.join(trained_model_location,"meta_scalar.joblib"))
    # Compatibility: older or differently-pickled MinMaxScaler objects may not
    # have the `clip` attribute (added in newer scikit-learn versions). Ensure
    # the attribute exists to avoid AttributeError during transform.
    if not hasattr(meta_scalar, 'clip'):
        # default to False which matches sklearn's default behavior
        try:
            setattr(meta_scalar, 'clip', False)
        except Exception:
            # If we can't set it for some reason, continue and let transform
            # raise a more informative error if needed.
            pass
    # generate meta features         
    meta_X, _ = generate_meta_features(X)
    # Check and adjust length of meta_vec to 200
    if len(meta_X) < 200:
        # Pad with zeros if meta_vec is too short
        meta_X.extend([0] * (200 - len(meta_X)))
    meta_X = np.nan_to_num(meta_X,nan=0.0, posinf=0.0, neginf=0.0)
    meta_X = meta_scalar.transform(np.asarray(meta_X).reshape(1, -1)).astype(float)
    # Convert to torch tensor for model input and ensure correct shape/dtype
    try:
        meta_X = torch.tensor(meta_X, dtype=torch.float32)
    except Exception:
        # Fallback: ensure numpy array then convert
        meta_X = torch.tensor(np.asarray(meta_X), dtype=torch.float32)


    # FMMS model parameters
    # The FMMS model expects input of the meta-feature vector size (e.g. 200),
    # not the original time-series feature count. Use the second dimension of
    # `meta_X` (after transform) as feature_size so it matches the checkpoint.
    meta_feat_dim = meta_X.shape[1] if hasattr(meta_X, 'shape') and len(meta_X.shape) > 1 else meta_X.shape[0]
    print(f"Meta-feature dim: {meta_feat_dim}, time-series feature dim: {X.shape[1]}")

    rate, data_name = config.get_rate()
    ytrain, ytest, xtrain, xtest = utils.get_data2(rate)
    model_size = ytrain.shape[1]

    params = {
        'embedding_size': 4,
        'feature_size':  int(meta_feat_dim),
        'model_size': model_size,  # 动态获取model_size而不是硬编码
        'FM': True,
        'DNN': False,
        'layer_size': 3,
        'hiddensize': 64
    }
    opt = 'adam'
    l = 'cos'
    train_params = {
        'batch': 8,
        'lr': 0.005,
        'epoch': 50,
        'opt': {'adam': torch.optim.Adam}[opt],
        'optname': opt,
        'loss': {'cos': utils.cos_loss}[l],
        'lossname': l,
    }
    path = config.get_para(train_params, params)
    print(f"Model Path: {path}")
    # Load FMMS model
    # Determine model identifier `txt`: allow override via argv[2], otherwise use data_name
    if len(sys.argv) >= 3:
        txt = sys.argv[2]
    else:
        txt = 0

    # Build model path and check existence
    model_filename = "FMMS%s_%s.pt" % (path, txt)
    model_path = os.path.join("models", model_filename)
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Make sure the model file exists or provide the correct identifier as a second argument.")
        sys.exit(1)

    fmms = FMMS(**params)
    # Load model to CPU by default to avoid device issues
    state = torch.load(model_path, map_location="cpu")
    fmms.load_state_dict(state)
    fmms.eval()

    # Ensure input has batch dimension
    if meta_X.ndim == 1:
        meta_X = meta_X.unsqueeze(0)

    with torch.no_grad():
        ypred = fmms(meta_X).detach().numpy()
    fmms_df = pd.read_csv("/home/yyy/TSC/TSClean/AutoClean/Error_Detection/FMMS/FMMS_ypred_0_ts_test_individual.csv")
    # Identify columns containing model scores (excluding the "Data" column)
    model_columns = fmms_df.columns.drop("Data")
    
    # Remove "Runtime" column as it's not an actual model
    model_columns = model_columns.drop("Runtime") if "Runtime" in model_columns else model_columns

    # Convert predictions to a pandas Series indexed by model column names.
    # Handle possible length mismatches by truncating or padding with -inf.
    ypred_arr = ypred.ravel()
    if ypred_arr.shape[0] != len(model_columns):
        print(f"Warning: number of predictions ({ypred_arr.shape[0]}) != number of model columns ({len(model_columns)}). Aligning lengths.")

    # Align lengths
    if ypred_arr.shape[0] < len(model_columns):
        pad = np.full(len(model_columns) - ypred_arr.shape[0], -np.inf, dtype=ypred_arr.dtype)
        ypred_aligned = np.concatenate([ypred_arr, pad])
    else:
        ypred_aligned = ypred_arr[:len(model_columns)]

    preds = pd.Series(ypred_aligned, index=model_columns)
    print("FMMS predictions:\n", preds)
    # Show top predictions and their scores for debugging
    top_3 = preds.sort_values(ascending=False).index[:3].tolist()
    print("top 3 models:")
    print(top_3)

    
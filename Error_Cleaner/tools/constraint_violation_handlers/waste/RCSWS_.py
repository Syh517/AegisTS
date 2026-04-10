import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union
import warnings

class RCSWS:
    """
    Python port of Java RCSWS algorithm for trajectory cleaning based on speed constraints.
    Supports multi-column DataFrame input, repairs only spatial (x,y) columns.
    """
    
    def __init__(self, smax: float, space_columns: List[str] = ['x', 'y']):
        """
        :param smax: Maximum allowed speed (unit: distance_unit / time_unit)
        :param space_columns: Names of spatial coordinate columns, e.g., ['x', 'y']
        """
        self.smax = smax
        self.space_columns = space_columns  # e.g., ['x', 'y']

    @staticmethod
    def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def judge_speed_constrain(self, tp1: pd.Series, tp2: pd.Series) -> bool:
        """Check if movement from tp1 to tp2 violates max speed constraint."""
        pos1 = tp1[self.space_columns].values.astype(float)
        pos2 = tp2[self.space_columns].values.astype(float)
        t1 = tp1['timestamp']
        t2 = tp2['timestamp']
        distance = self.calculate_distance(pos1, pos2)
        max_range = self.smax * abs(t2 - t1)
        return distance <= max_range

    def judge_speed_constrain_point(self, tp1: pd.Series, p: np.ndarray, t: int) -> bool:
        """Check if point p at time t satisfies speed constraint from tp1."""
        pos1 = tp1[self.space_columns].values.astype(float)
        distance = self.calculate_distance(pos1, p)
        max_range = self.smax * abs(tp1['timestamp'] - t)
        return distance <= max_range

    def judge_intersection(self, tp1: pd.Series, tp2: pd.Series, t: int) -> bool:
        """Check if two circles (around tp1 and tp2 at time t) intersect."""
        pos1 = tp1[self.space_columns].values.astype(float)
        pos2 = tp2[self.space_columns].values.astype(float)
        t1 = tp1['timestamp']
        t2 = tp2['timestamp']
        
        d = self.calculate_distance(pos1, pos2)
        r1 = self.smax * abs(t1 - t)
        r2 = self.smax * abs(t2 - t)
        
        # Check intersection: |r1 - r2| < d < r1 + r2
        eps = 1e-6
        return (d < r1 + r2 + eps) and (d > abs(r1 - r2) + eps)

    def calculate_line_circle_intersection(self, p_start: np.ndarray, p_dir: np.ndarray,
                                           center: np.ndarray, radius: float) -> Optional[np.ndarray]:
        """
        Calculate intersection between a line (defined by start point and direction vector)
        and a circle (center, radius).
        Returns: array of shape (n_points, 2), or None if no intersection.
        """
        # Line: P = p_start + t * p_dir
        dx, dy = p_dir
        if np.isclose(dx, 0) and np.isclose(dy, 0):
            return None
        
        fx, fy = p_start - center
        a = dx*dx + dy*dy
        b = 2 * (fx*dx + fy*dy)
        c = fx*fx + fy*fy - radius*radius
        
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2*a)
        t2 = (-b - sqrt_disc) / (2*a)
        
        p1 = p_start + t1 * p_dir
        p2 = p_start + t2 * p_dir
        return np.array([p1, p2])

    def calculate_intersection_points(self, center1: np.ndarray, r1: float,
                                      center2: np.ndarray, r2: float) -> Optional[np.ndarray]:
        """
        Calculate intersection points of two circles.
        Returns: (2, 2) array of [[x1,y1], [x2,y2]], or None.
        """
        d = self.calculate_distance(center1, center2)
        if d > r1 + r2 or d < abs(r1 - r2):
            return None  # No intersection
        
        # Use geometric method
        a = (r1*r1 - r2*r2 + d*d) / (2*d)
        h = np.sqrt(max(0, r1*r1 - a*a))  # Avoid negative due to precision

        # Unit vector from center1 to center2
        ex = (center2 - center1) / d
        # Perpendicular vector
        e_perp = np.array([-ex[1], ex[0]])

        p1 = center1 + a * ex + h * e_perp
        p2 = center1 + a * ex - h * e_perp
        return np.array([p1, p2])

    def clean(self, df: pd.DataFrame, wSize: int = 50) -> pd.DataFrame:
        """
        Main cleaning function. Repairs spatial coordinates in the trajectory.
        
        :param df: Input DataFrame with 'timestamp' and spatial columns (e.g., 'x', 'y')
        :param wSize: Window size (currently unused; kept for interface compatibility)
        :return: DataFrame with repaired columns and 'is_fixed' flag
        """
        # Ensure sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True).copy()
        
        n = len(df)
        if n < 3:
            df[[f"{col}_repaired" for col in self.space_columns]] = df[self.space_columns]
            df['is_fixed'] = False
            return df

        # Add repaired columns
        for col in self.space_columns:
            df[f"{col}_repaired"] = df[col]

        df['is_fixed'] = False

        # Main loop: fix points from index 1 to n-2
        for i in range(1, n - 1):
            pre_point = df.iloc[i - 1]      # tp_{i-1}
            key_point = df.iloc[i]          # tp_i (to be fixed)
            temp_index = i + 1
            temp_point = df.iloc[temp_index]  # next available point

            # Extract current position of key_point (before repair)
            curr_pos = np.array([key_point[col] for col in self.space_columns], dtype=float)
            key_timestamp = key_point['timestamp']

            # Check if pre_point → key_point violates speed constraint
            if not self.judge_speed_constrain(pre_point, key_point):
                # Find first temp_point such that circles intersect at key_timestamp
                flag = True
                while temp_index < n - 1 and not self.judge_intersection(pre_point, temp_point, key_timestamp):
                    temp_index += 1
                    if temp_index < n:
                        temp_point = df.iloc[temp_index]
                    else:
                        flag = False
                        break

                # Get positions
                pre_pos = np.array([pre_point[f"{col}_repaired"] for col in self.space_columns], dtype=float)
                kp_pos = np.array([key_point[col] for col in self.space_columns], dtype=float)

                # Vector from pre_point to key_point (direction)
                direction_vec = kp_pos - pre_pos

                # Radius for pre_point at key_timestamp
                r_pre = self.smax * abs(pre_point['timestamp'] - key_timestamp)

                # Compute intersection of line (pre → key) with circle around pre_point
                intersections = self.calculate_line_circle_intersection(
                    p_start=pre_pos,
                    p_dir=direction_vec,
                    center=pre_pos,
                    radius=r_pre
                )

                if intersections is None or len(intersections) == 0:
                    # Fallback: move toward pre_point within valid range
                    unit_dir = direction_vec / (np.linalg.norm(direction_vec) + 1e-8)
                    candidate_p1 = pre_pos + unit_dir * r_pre
                else:
                    # Choose the intersection point closer to original key_point
                    dist1 = self.calculate_distance(intersections[0], kp_pos)
                    dist2 = self.calculate_distance(intersections[1], kp_pos)
                    candidate_p1 = intersections[0] if dist1 <= dist2 else intersections[1]

                # Check if candidate_p1 satisfies constraint with temp_point
                if flag and temp_index < n:
                    satisfies_temp = self.judge_speed_constrain_point(temp_point, candidate_p1, key_timestamp)
                else:
                    satisfies_temp = False

                if satisfies_temp or not flag:
                    # Case 1: Fix using line-circle intersection
                    repair_pos = candidate_p1
                else:
                    # Case 2/3: Use intersection of two circles (pre_point and temp_point)
                    r_temp = self.smax * abs(temp_point['timestamp'] - key_timestamp)
                    temp_pos = np.array([temp_point[f"{col}_repaired"] for col in self.space_columns], dtype=float)
                    
                    circle_intersections = self.calculate_intersection_points(
                        center1=pre_pos, r1=r_pre,
                        center2=temp_pos, r2=r_temp
                    )
                    
                    if circle_intersections is not None and len(circle_intersections) > 0:
                        # Choose the intersection closer to original key_point
                        dist1 = self.calculate_distance(circle_intersections[0], kp_pos)
                        dist2 = self.calculate_distance(circle_intersections[1], kp_pos)
                        repair_pos = circle_intersections[0] if dist1 <= dist2 else circle_intersections[1]
                    else:
                        # Fallback to line-circle if two-circle fails
                        repair_pos = candidate_p1
                # Apply repair
                for idx, col in enumerate(self.space_columns):
                    df.loc[i, f"{col}_repaired"] = repair_pos[idx]
                df.loc[i, 'is_fixed'] = True

            else:
                # No violation, keep original value (already copied)
                pass

        return df
    


if __name__ == '__main__':
    # ====== 构造含轨迹跳点的数据 ======
    df = pd.DataFrame({
        "timestamp": np.arange(10),
        "x": [0,1,2,3,4,5,20,6,7,8],  # 第6个点跳了(从5直接跳到20)
        "y": [0,0,0,0,0,0,20,0,0,0]
    })

    print("=== 原始数据 ===")
    print(df)

    # ====== 调用 RCSWS ======
    rc = RCSWS(smax=3.0, space_columns=['x','y'])
    df_clean = rc.clean(df)

    print("\n=== 修复后的数据 ===")
    print(df_clean)

    # 查看哪些点被修复
    print("\n=== 被修复点索引 ===")
    print(df_clean[df_clean['is_fixed'] == True])
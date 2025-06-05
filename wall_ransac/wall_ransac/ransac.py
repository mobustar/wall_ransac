import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray, Header
import numpy as np
import math
import random
from typing import List, Tuple, Optional

class WallDetectionNode(Node):
    def __init__(self):
        super().__init__('ransac_node')
        
        # パラメータ設定
        self.declare_parameter('min_distance_threshold', 0.1)
        self.declare_parameter('max_distance_threshold', 5.0)
        self.declare_parameter('ransac_iterations', 1000)
        self.declare_parameter('ransac_threshold', 0.05)
        self.declare_parameter('min_points_for_wall', 10)
        self.declare_parameter('wall_distance_threshold', 1.0)  # 壁との距離閾値
        
        # パラメータ取得
        self.min_distance = self.get_parameter('min_distance_threshold').get_parameter_value().double_value
        self.max_distance = self.get_parameter('max_distance_threshold').get_parameter_value().double_value
        self.ransac_iterations = self.get_parameter('ransac_iterations').get_parameter_value().integer_value
        self.ransac_threshold = self.get_parameter('ransac_threshold').get_parameter_value().double_value
        self.min_points_for_wall = self.get_parameter('min_points_for_wall').get_parameter_value().integer_value
        self.wall_distance_threshold = self.get_parameter('wall_distance_threshold').get_parameter_value().double_value
        
        # Subscriber
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        # Publishers
        self.wall_data_publisher = self.create_publisher(
            Float32MultiArray,
            '/ransac_data',
            10
        )
        
        self.get_logger().info('壁検知ノードが開始されました')
        self.get_logger().info(f'RANSACパラメータ: iterations={self.ransac_iterations}, threshold={self.ransac_threshold}')

    def scan_callback(self, msg: LaserScan):
        """Lidarスキャンデータのコールバック関数"""
        try:
            # スキャンデータを直交座標系に変換
            points = self.convert_scan_to_points(msg)
            
            if len(points) < self.min_points_for_wall:
                self.get_logger().warn('十分なポイントが得られませんでした')
                return
            
            # RANSACを使用して壁を検出
            walls = self.detect_walls_ransac(points)
            
            # 検出された壁に基づいて処理を実行
            self.process_wall_data(walls, msg.header)
            
        except Exception as e:
            self.get_logger().error(f'スキャンデータ処理中にエラー: {str(e)}')

    def convert_scan_to_points(self, scan: LaserScan) -> List[Tuple[float, float]]:
        """LaserScanデータを(x, y)座標点のリストに変換"""
        points = []
        
        for i, distance in enumerate(scan.ranges):
            # 無効な距離データをフィルタリング
            if (math.isnan(distance) or math.isinf(distance) or 
                distance < self.min_distance or distance > self.max_distance):
                continue
            
            # 角度計算
            angle = scan.angle_min + i * scan.angle_increment
            
            # 直交座標系に変換
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            
            points.append((x, y))
        
        return points

    def detect_walls_ransac(self, points: List[Tuple[float, float]]) -> List[dict]:
        """RANSACアルゴリズムを使用して壁を検出"""
        walls = []
        remaining_points = points.copy()
        
        # 複数の壁を検出するためのループ
        wall_id = 0
        while len(remaining_points) >= self.min_points_for_wall and wall_id < 5:  # 最大5つの壁まで検出
            wall = self.ransac_line_fitting(remaining_points)
            
            if wall is not None:
                walls.append({
                    'id': wall_id,
                    'line_params': wall['line_params'],
                    'inliers': wall['inliers'],
                    'distance_to_origin': wall['distance_to_origin']
                })
                
                # インライアポイントを除去
                remaining_points = [p for p in remaining_points if p not in wall['inliers']]
                wall_id += 1
                
                self.get_logger().info(f'壁 {wall_id} を検出: 距離={wall["distance_to_origin"]:.3f}m, ポイント数={len(wall["inliers"])}')
            else:
                break
        
        return walls

    def ransac_line_fitting(self, points: List[Tuple[float, float]]) -> Optional[dict]:
        """RANSACアルゴリズムによる直線フィッティング"""
        if len(points) < 2:
            return None
        
        best_inliers = []
        best_line_params = None
        best_distance_to_origin = float('inf')
        
        for _ in range(self.ransac_iterations):
            # ランダムに2点を選択
            sample_points = random.sample(points, 2)
            p1, p2 = sample_points
            
            # 直線パラメータを計算 (ax + by + c = 0の形式)
            line_params = self.calculate_line_params(p1, p2)
            if line_params is None:
                continue
            
            a, b, c = line_params
            
            # インライアを計算
            inliers = []
            for point in points:
                distance = abs(a * point[0] + b * point[1] + c) / math.sqrt(a**2 + b**2)
                if distance < self.ransac_threshold:
                    inliers.append(point)
            
            # 最良の結果を更新
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_line_params = line_params
                # 原点から直線までの距離を計算
                best_distance_to_origin = abs(c) / math.sqrt(a**2 + b**2)
        
        # 十分なインライアがある場合のみ壁として認識
        if len(best_inliers) >= self.min_points_for_wall:
            return {
                'line_params': best_line_params,
                'inliers': best_inliers,
                'distance_to_origin': best_distance_to_origin
            }
        
        return None

    def calculate_line_params(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Optional[Tuple[float, float, float]]:
        """2点から直線パラメータ(a, b, c)を計算"""
        x1, y1 = p1
        x2, y2 = p2
        
        # 同じ点の場合は無効
        if abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6:
            return None
        
        # ax + by + c = 0の形式で計算
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        # 正規化
        norm = math.sqrt(a**2 + b**2)
        if norm < 1e-6:
            return None
        
        return (a/norm, b/norm, c/norm)

    def process_wall_data(self, walls: List[dict], header: Header):
        """検出された壁の情報に基づいてデータ処理を実行"""
        if not walls:
            self.get_logger().info('壁が検出されませんでした')
            return
        
        # 壁データメッセージを作成
        wall_msg = Float32MultiArray()
        wall_msg.data = []
        
        for wall in walls:
            wall_id = wall['id']
            distance = wall['distance_to_origin']
            num_points = len(wall['inliers'])
            
            # 距離に応じた処理
            if distance < self.wall_distance_threshold:
                self.get_logger().warn(f'壁 {wall_id}: 近接警告! 距離={distance:.3f}m')
                status = 1.0  # 警告状態
            else:
                self.get_logger().info(f'壁 {wall_id}: 安全距離 距離={distance:.3f}m')
                status = 0.0  # 正常状態
            
            # データ配列に追加 [wall_id, distance, num_points, status]
            wall_msg.data.extend([float(wall_id), distance, float(num_points), status])
        
        # データを発行
        self.wall_data_publisher.publish(wall_msg)
        
        # 最も近い壁の情報をログ出力
        closest_wall = min(walls, key=lambda w: w['distance_to_origin'])
        self.get_logger().info(f'最も近い壁: ID={closest_wall["id"]}, 距離={closest_wall["distance_to_origin"]:.3f}m')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        ransac_node = WallDetectionNode()
        rclpy.spin(ransac_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'ノード実行中にエラーが発生しました: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

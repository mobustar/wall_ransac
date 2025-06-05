import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray, Header
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
import random
from typing import List, Tuple, Optional
import struct

class WallDetectionNode(Node):
    def __init__(self):
        super().__init__('ransac_node')
        
        # パラメータ設定
        self.declare_parameter('min_distance_threshold', 0.1)
        self.declare_parameter('max_distance_threshold', 5.0)
        self.declare_parameter('ransac_iterations', 1000)
        self.declare_parameter('ransac_threshold', 0.05)
        self.declare_parameter('min_points_for_wall', 10)
        self.declare_parameter('wall_distance_threshold', 1.0)
        
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
        
        # RViz2用のPublishers
        self.wall_markers_publisher = self.create_publisher(
            MarkerArray,
            '/wall_markers',
            10
        )
        
        self.wall_points_publisher = self.create_publisher(
            PointCloud2,
            '/wall_points',
            10
        )
        
        self.filtered_scan_publisher = self.create_publisher(
            LaserScan,
            '/filtered_scan',
            10
        )
        
        self.get_logger().info('壁検知ノードが開始されました (RViz2対応)')
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
            
            # RViz2用の視覚化データを発行
            self.publish_visualization_data(walls, points, msg)
            
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
        while len(remaining_points) >= self.min_points_for_wall and wall_id < 5:
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
            
            # 直線パラメータを計算
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

    def publish_visualization_data(self, walls: List[dict], all_points: List[Tuple[float, float]], scan_msg: LaserScan):
        """RViz2用の視覚化データを発行"""
        # 壁のマーカーを作成
        self.publish_wall_markers(walls, scan_msg.header)
        
        # 壁のポイントクラウドを作成
        self.publish_wall_points(walls, scan_msg.header)
        
        # フィルタリングされたスキャンデータを発行
        self.publish_filtered_scan(walls, all_points, scan_msg)

    def publish_wall_markers(self, walls: List[dict], header: Header):
        """壁を直線として表示するマーカーを発行"""
        marker_array = MarkerArray()
        
        for wall in walls:
            # 直線マーカーを作成
            line_marker = Marker()
            line_marker.header = header
            line_marker.ns = "walls"
            line_marker.id = wall['id']
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            
            # 直線の色を距離に応じて設定
            distance = wall['distance_to_origin']
            if distance < self.wall_distance_threshold:
                line_marker.color.r = 1.0  # 赤色（警告）
                line_marker.color.g = 0.0
                line_marker.color.b = 0.0
            else:
                line_marker.color.r = 0.0
                line_marker.color.g = 1.0  # 緑色（安全）
                line_marker.color.b = 0.0
            
            line_marker.color.a = 0.8
            line_marker.scale.x = 0.05  # 線の太さ
            
            # 直線の両端点を計算
            a, b, c = wall['line_params']
            
            # インライア点の範囲を計算
            inliers = wall['inliers']
            if len(inliers) >= 2:
                # インライア点の両端を見つける
                x_coords = [p[0] for p in inliers]
                y_coords = [p[1] for p in inliers]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # 直線上の点を計算
                if abs(b) > abs(a):  # y = -(ax + c)/b
                    start_x = x_min - 0.2
                    start_y = -(a * start_x + c) / b
                    end_x = x_max + 0.2
                    end_y = -(a * end_x + c) / b
                else:  # x = -(by + c)/a
                    start_y = y_min - 0.2
                    start_x = -(b * start_y + c) / a
                    end_y = y_max + 0.2
                    end_x = -(b * end_y + c) / a
                
                # 点を追加
                start_point = Point()
                start_point.x, start_point.y, start_point.z = float(start_x), float(start_y), 0.0
                
                end_point = Point()
                end_point.x, end_point.y, end_point.z = float(end_x), float(end_y), 0.0
                
                line_marker.points = [start_point, end_point]
                marker_array.markers.append(line_marker)
            
            # 距離テキストマーカーを作成
            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "wall_distances"
            text_marker.id = wall['id'] + 100
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # テキストの位置（壁の中心付近）
            center_x = sum(p[0] for p in inliers) / len(inliers)
            center_y = sum(p[1] for p in inliers) / len(inliers)
            text_marker.pose.position.x = float(center_x)
            text_marker.pose.position.y = float(center_y)
            text_marker.pose.position.z = 0.2
            
            # テキストの内容と色
            text_marker.text = f"Wall {wall['id']}\n{distance:.2f}m"
            text_marker.scale.z = 0.2  # テキストサイズ
            
            if distance < self.wall_distance_threshold:
                text_marker.color.r = 1.0
                text_marker.color.g = 0.0
                text_marker.color.b = 0.0
            else:
                text_marker.color.r = 0.0
                text_marker.color.g = 1.0
                text_marker.color.b = 0.0
            text_marker.color.a = 1.0
            
            marker_array.markers.append(text_marker)
        
        self.wall_markers_publisher.publish(marker_array)

    def publish_wall_points(self, walls: List[dict], header: Header):
        """壁のポイントをPointCloud2として発行"""
        # PointCloud2メッセージを作成
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        
        # フィールド定義
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16
        
        # ポイントデータを準備
        points_data = []
        colors = [
            0xFF0000,  # 赤
            0x00FF00,  # 緑
            0x0000FF,  # 青
            0xFFFF00,  # 黄
            0xFF00FF,  # マゼンタ
        ]
        
        for wall in walls:
            color = colors[wall['id'] % len(colors)]
            for point in wall['inliers']:
                x, y = point
                z = 0.0
                points_data.append(struct.pack('fffI', x, y, z, color))
        
        cloud_msg.width = len(points_data)
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.data = b''.join(points_data)
        
        self.wall_points_publisher.publish(cloud_msg)

    def publish_filtered_scan(self, walls: List[dict], all_points: List[Tuple[float, float]], original_scan: LaserScan):
        """壁以外のポイントを含むフィルタリングされたスキャンを発行"""
        # 壁のポイントを集める
        wall_points = set()
        for wall in walls:
            for point in wall['inliers']:
                wall_points.add(point)
        
        # フィルタリングされたスキャンメッセージを作成
        filtered_scan = LaserScan()
        filtered_scan.header = original_scan.header
        filtered_scan.angle_min = original_scan.angle_min
        filtered_scan.angle_max = original_scan.angle_max
        filtered_scan.angle_increment = original_scan.angle_increment
        filtered_scan.time_increment = original_scan.time_increment
        filtered_scan.scan_time = original_scan.scan_time
        filtered_scan.range_min = original_scan.range_min
        filtered_scan.range_max = original_scan.range_max
        
        # 壁以外のポイントのみを含む新しいrangesを作成
        filtered_ranges = []
        for i, distance in enumerate(original_scan.ranges):
            if (math.isnan(distance) or math.isinf(distance) or 
                distance < self.min_distance or distance > self.max_distance):
                filtered_ranges.append(float('nan'))
                continue
            
            angle = original_scan.angle_min + i * original_scan.angle_increment
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            
            # 壁のポイントの場合はnanにする
            if (x, y) in wall_points:
                filtered_ranges.append(float('nan'))
            else:
                filtered_ranges.append(distance)
        
        filtered_scan.ranges = filtered_ranges
        self.filtered_scan_publisher.publish(filtered_scan)


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

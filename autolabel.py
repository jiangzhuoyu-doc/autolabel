import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from autolabel_tools.picture import *
from autolabel_tools.video import *
from autolabel_tools.camera import *
from auto_detect.get_box import *
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QRadioButton, QButtonGroup, QComboBox, QDialogButtonBox, QHeaderView
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QGraphicsScene, QGraphicsView, QFileDialog, QGraphicsEllipseItem, QDialog, QLineEdit, QHBoxLayout, QTableWidget, QTableWidgetItem
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QMouseEvent
from PyQt6.QtCore import Qt

class PointObject:
    def __init__(self, x, y, label="", property=""):
        self.x = x
        self.y = y
        self.label = label
        self.property = property

class CustomHeaderView(QHeaderView):
    def __init__(self, orientation, main_window, parent=None):
        super().__init__(orientation, parent)
        self.main_window = main_window  
        self.setSectionsClickable(True)  
        self.setSectionsMovable(True)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.logicalIndexAt(event.pos())
            if index >= 0:
                self.handleHeaderClick(index)
        super().mousePressEvent(event)

    def handleHeaderClick(self, index):
        if self.orientation() == Qt.Orientation.Vertical:  
            row = index
            self.main_window.set_if_delete(row)  

class LabelInputDialog(QDialog):
    def __init__(self, parent=None, show_buttons=False):
        super().__init__(parent)
        self.setWindowTitle("输入标签")
        self.label_edit = QLineEdit()
        self.ok_button = QPushButton("确认")
        self.ok_button.clicked.connect(self.accept)
        
        self.positive_button = QRadioButton("Positive")
        self.negative_button = QRadioButton("Negative")
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.positive_button)
        self.button_group.addButton(self.negative_button)

        self.show_buttons = show_buttons
        self.buttons_layout = QHBoxLayout()
        if self.show_buttons:
            self.buttons_layout.addWidget(self.positive_button)
            self.buttons_layout.addWidget(self.negative_button)

        layout = QVBoxLayout()
        layout.addWidget(self.label_edit)
        layout.addLayout(self.buttons_layout)
        layout.addWidget(self.ok_button)
        self.setLayout(layout)
        
    def get_label_text(self):
        return self.label_edit.text()
    
    def get_selected_button_text(self):
        if self.show_buttons and self.button_group.checkedButton() is not None:
            return self.button_group.checkedButton().text()
        else:
            return ""

class CameraSelectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择相机类型")

        self.camera_type_combo = QComboBox()
        self.camera_type_combo.addItem("realsense")
        self.camera_type_combo.addItem("other")

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("请选择相机类型:"))
        layout.addWidget(self.camera_type_combo)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def get_selected_camera(self):
        return self.camera_type_combo.currentText()

class ImageAnnotationGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image = None
        self.video_save_dir = None
        self.camera = None
        self.camera_type = None
        self.annotated_ategories = None
        self.save_type = None
        self.data_save_dir = None
        self.if_delete = False
        self.delete_item = None
        self.points = []
        self.point_items = []  
        self.point_objects = []  
        self.point_labels = []

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("AutoLabel")
        self.setGeometry(100, 100, 1600, 900) 

        # 创建按钮
        self.load_image_button = QPushButton("加载图像")
        self.load_image_button.setMaximumWidth(200)
        self.load_image_button.clicked.connect(self.load_image)

        self.load_video_button = QPushButton("加载视频")
        self.load_video_button.setMaximumWidth(200)
        self.load_video_button.clicked.connect(self.load_video)

        self.load_video_stream_button = QPushButton("加载视频流")
        self.load_video_stream_button.setMaximumWidth(200)
        self.load_video_stream_button.clicked.connect(self.load_video_stream)

        self.exit_button = QPushButton("确认退出")
        self.exit_button.setEnabled(True)  
        self.exit_button.setMaximumWidth(200)
        self.exit_button.clicked.connect(self.confirm_exit)

        self.set_positive_button = QPushButton("设为 Positive")
        self.set_positive_button.setEnabled(False)
        self.set_positive_button.setMaximumWidth(200)
        self.set_positive_button.clicked.connect(self.set_positive_callback)

        self.set_negative_button = QPushButton("设为 Negative")
        self.set_negative_button.setEnabled(False)
        self.set_negative_button.setMaximumWidth(200)
        self.set_negative_button.clicked.connect(self.set_negative_callback)

        self.set_label_button = QPushButton("设置标签")
        self.set_label_button.setEnabled(False)
        self.set_label_button.setMaximumWidth(200)
        self.set_label_button.clicked.connect(self.set_label_callback)

        self.delete_button = QPushButton("删除所选项")
        self.delete_button.setEnabled(True)
        self.delete_button.setMaximumWidth(200)
        self.delete_button.clicked.connect(self.delete_callback)

        self.autolabel_button = QPushButton("自动标注")
        self.autolabel_button.setEnabled(False)
        self.autolabel_button.setMaximumWidth(200)
        self.autolabel_button.clicked.connect(self.autolabel)

        self.save_button = QComboBox()
        self.save_button.addItem("box")
        self.save_button.addItem("mask")
        self.save_button.setMaximumWidth(200)
        self.save_button.setCurrentIndex(1) 
        
        # 创建布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        # 创建右侧区域布局
        self.right_layout = QVBoxLayout()
        self.point_table = QTableWidget()
        self.point_table.setColumnCount(4)  
        self.point_table.setMaximumWidth(400)
        self.point_table.setHorizontalHeaderLabels(["X", "Y", "Label", "Property"])
        self.vertical_header = CustomHeaderView(Qt.Orientation.Vertical, self, self.point_table)
        self.point_table.setVerticalHeader(self.vertical_header)
        self.point_table.cellClicked.connect(self.edit_point)
        self.point_table.cellClicked.connect(self.set_if_delete)


        self.right_layout.addWidget(self.point_table)

        self.main_layout.addWidget(self.graphics_view)
        self.main_layout.addLayout(self.right_layout)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.load_image_button)
        button_layout.addWidget(self.load_video_button)
        button_layout.addWidget(self.load_video_stream_button)
        button_layout.addWidget(self.set_positive_button)
        button_layout.addWidget(self.set_negative_button)
        button_layout.addWidget(self.set_label_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.autolabel_button)
        button_layout.addWidget(self.exit_button)
        self.main_layout.addLayout(button_layout)

    def get_init_points(self):
        box, classes = get_box2(frame=self.image)
        points = get_centrel_point(boxes=box)
        for i in range(0, len(points)):
            x = points[i][0]
            y = points[i][1]
            self.points.append((int(x), int(y)))
            self.point_objects.append(PointObject(int(x), int(y)))
            self.set_point_property(property = "Positive")
            self.set_point_label(label=classes[i])
        
        self.update_image()
        self.draw_points()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Image files (*.jpg *.png)")
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            self.clear_points()
            self.get_init_points()
            self.set_positive_button.setEnabled(True)
            self.set_negative_button.setEnabled(True)
            self.set_label_button.setEnabled(True)
            self.exit_button.setEnabled(True)
            self.autolabel_button.setEnabled(True)
            self.annotated_ategories = "image" 

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video files (*.mp4 *.avi)")
        if file_path:
            self.video_save_dir = QFileDialog.getExistingDirectory(self, "选择保存路径")
            if self.video_save_dir:
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    print("无法打开视频文件")
                    return
                
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                success, frame = self.cap.read()
                if success:
                    self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    for i in range(frame_count):
                        success, frame = self.cap.read()
                        if not success:
                            break
                        frame_path = f"{self.video_save_dir}/{i}.jpg"
                        cv2.imwrite(frame_path, frame)
                    
                    self.clear_points()
                    self.get_init_points()
                    self.set_positive_button.setEnabled(True)
                    self.set_negative_button.setEnabled(True)
                    self.set_label_button.setEnabled(True)
                    self.exit_button.setEnabled(True)
                    self.autolabel_button.setEnabled(True)
                    self.annotated_ategories = "video" 
            else:
                print("无法打开保存路径")
        else:
            print("重新选择视频路径")

    def load_video_stream(self):
        dialog = CameraSelectDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_camera = dialog.get_selected_camera()
            if selected_camera == "realsense":
                print("选择了 Realsense 摄像机")
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) 
                pipeline.start(config)
                self.camera = pipeline
                self.camera_type = "realsense"
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    print("相机加载失败")
                else:
                    frame = np.asanyarray(color_frame.get_data())
                    self.image = frame
                    self.clear_points()
                    self.get_init_points()
                    self.set_positive_button.setEnabled(True)
                    self.set_negative_button.setEnabled(True)
                    self.set_label_button.setEnabled(True)
                    self.exit_button.setEnabled(True) 
                    self.autolabel_button.setEnabled(True)
                    self.annotated_ategories = "camera"


            elif selected_camera == "other":
                print("选择了其他类型的摄像机")
                # 在这里添加打开其他类型摄像机的代码

    def clear_points(self):
        self.points.clear()
        self.point_labels.clear()
        for item in self.point_items:
            self.scene.removeItem(item)
        self.point_items.clear()
        self.point_objects.clear()
        self.update_point_table()

    def update_image(self):
        if self.image is not None:
            height, width, channel = self.image.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.scene.clear()
            self.scene.addPixmap(pixmap)
            self.graphics_view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_positive_callback(self):
        if self.points:
            self.set_point_property("Positive")
            print("设置点属性为 Positive")

    def set_negative_callback(self):
        if self.points:
            self.set_point_property("Negative")
            print("设置点属性为 Negative")

    def set_label_callback(self):
        if self.points:
            dialog = LabelInputDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                label = dialog.get_label_text()
                self.set_point_label(label)
                print(f"设置点的标签为: {label}")

    def set_point_label(self, label):
        for point_obj in self.point_objects:
            if point_obj.x == self.points[-1][0] and point_obj.y == self.points[-1][1]:
                point_obj.label = label
                break
        self.update_point_table()

    def set_point_property(self, property):
        for point_obj in self.point_objects:
            if point_obj.x == self.points[-1][0] and point_obj.y == self.points[-1][1]:
                point_obj.property = property
                break
        self.update_point_table()

    def set_if_delete(self, row):
        if 0 <= row <=  len(self.point_objects):
            self.if_delete = True
            self.delete_item = row
    
    def delete_callback(self):
        if self.if_delete == True:    
            del self.point_objects[self.delete_item]
            del self.points[self.delete_item]
            self.if_delete = False
            self.update_point_table()
            self.draw_points()
        else:
            print("请选择需要删除项")

    def edit_point(self, row, column):
        if column == 2 or column == 3:  
            selected_point = self.point_objects[row]
            dialog = LabelInputDialog(self)
            if column == 2:
                dialog.label_edit.setText(selected_point.label)
            elif column == 3:
                dialog = LabelInputDialog(self, show_buttons=True)
                if selected_point.property == "Positive":
                    dialog.positive_button.setChecked(True)
                elif selected_point.property == "Negative":
                    dialog.negative_button.setChecked(True)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                if column == 2:
                    new_text = dialog.get_label_text()
                    selected_point.label = new_text
                elif column == 3:
                    new_text = dialog.get_selected_button_text()
                    selected_point.property = new_text
                self.update_point_table()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.image is not None:
            pos = self.graphics_view.mapToScene(event.pos())
            x, y = pos.x(), pos.y()
            self.points.append((int(x), int(y)))
            self.point_objects.append(PointObject(int(x), int(y)))

            self.draw_points()

    def draw_points(self):
        for item in self.point_items:
            self.scene.removeItem(item)
        self.point_items.clear()

        for point in self.points:
            x, y = point
            ellipse = QGraphicsEllipseItem(x - 2, y - 2, 4, 4)
            ellipse.setPen(QPen(Qt.GlobalColor.red, 2))
            self.scene.addItem(ellipse)
            self.point_items.append(ellipse)

        self.graphics_view.setScene(self.scene)
        self.update_point_table()

    def update_point_table(self):
        self.point_table.setRowCount(len(self.point_objects))
        for i, point_obj in enumerate(self.point_objects):
            x_item = QTableWidgetItem(str(point_obj.x))
            y_item = QTableWidgetItem(str(point_obj.y))
            label_item = QTableWidgetItem(point_obj.label)
            property_item = QTableWidgetItem(point_obj.property)
            self.point_table.setItem(i, 0, x_item)
            self.point_table.setItem(i, 1, y_item)
            self.point_table.setItem(i, 2, label_item)
            self.point_table.setItem(i, 3, property_item)

    def confirm_exit(self):
        points, properties, labels = self.get_annotations()
        print("Points:", points)
        print("Properties:", properties)
        print("Labels:", labels)
        self.close()

    def get_annotations(self):
        points = [(point.x, point.y) for point in self.point_objects]
        properties = [point.property for point in self.point_objects]
        labels = [point.label for point in self.point_objects]
        return points, properties, labels
    
    def set_saveconfig(self):
        self.save_type = self.save_button.currentText()
        self.data_save_dir = QFileDialog.getExistingDirectory(self, "选择保存路径")

    
    def autolabel(self):
        self.set_saveconfig()

        if self.annotated_ategories == "image":
            print("功能在开发，敬请期待")
        elif self.annotated_ategories == "video":
            points, properties, labels = self.get_annotations()
            autolabel_video(video_dir=self.video_save_dir, first_frame=self.image, points=points, obj_labels=labels, 
                            obj_properties=properties, save_type=self.save_type, save_dir=self.data_save_dir)
        elif self.annotated_ategories == "camera":
            points, properties, labels = self.get_annotations()
            autolabel_camera(camera=self.camera, camera_type=self.camera_type, first_frame=self.image, 
                             points=points, obj_labels=labels, obj_properties=properties,
                             save_type = self.save_type, save_dir=self.data_save_dir)
        else:
            print("请先加载图像，视频或者相机")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ImageAnnotationGUI()
    gui.show()
    sys.exit(app.exec())

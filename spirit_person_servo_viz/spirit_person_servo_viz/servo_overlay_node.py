"""Draw the person servo's target on the Spirit EO video in Foxglove.

Runs on the ground, next to the video. Subscribes the servo's ServoState (bridged from
the aircraft's domain; ~200 bytes at the servo's state rate) and publishes
foxglove_msgs/ImageAnnotations: a box around the chosen person, with its state and
confidence, ONLY while the servo is engaged on a target (LOCKED, SERVOING, HOLD with
has_target). In every other state it publishes an empty annotation, so the box
disappears the moment servoing stops. Nothing is decoded or re-encoded and no second
image stream crosses the radio.

Coordinates: the servo decodes the same RTSP stream the video relay forwards (the
Jetson's sender_rtsp re-host) at full resolution and reports bbox_* in those pixels, so
the box lands on the matching pixels of /<robot>/<sensor>/video.

Timestamps: Foxglove pairs an annotation with the image whose header.stamp matches.
The video carries the aircraft's capture time (restamped by the relay), ServoState the
servo's own clock, so with stamp_from_video (default) each annotation takes the stamp of
the latest video frame received here: the box shows on the current frame, trailing the
servo by its processing delay. Set it false to keep ServoState's stamp (then turn off
annotation sync in the Foxglove Image panel).

In Foxglove: Image panel on /<robot>/<sensor>/video, enable the annotations topic.
"""

import rclpy
from foxglove_msgs.msg import (Color, CompressedVideo, ImageAnnotations, Point2,
                               PointsAnnotation, TextAnnotation)
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from spirit_person_servo_msgs.msg import ServoState

ENGAGED = {ServoState.STATE_LOCKED, ServoState.STATE_SERVOING, ServoState.STATE_HOLD}
STATE_NAMES = {ServoState.STATE_IDLE: 'IDLE', ServoState.STATE_SEARCHING: 'SEARCHING',
               ServoState.STATE_LOCKED: 'LOCKED', ServoState.STATE_SERVOING: 'SERVOING',
               ServoState.STATE_HOLD: 'HOLD', ServoState.STATE_LOST: 'LOST'}
# Green while actively driving, amber while settled in the deadband.
COLORS = {ServoState.STATE_SERVOING: (0.1, 0.9, 0.2), ServoState.STATE_LOCKED: (0.1, 0.9, 0.2),
          ServoState.STATE_HOLD: (1.0, 0.75, 0.1)}


class ServoOverlay(Node):
    def __init__(self):
        super().__init__('servo_overlay')
        robot = self.declare_parameter('robot', 'spiritnx3').value
        state_topic = self.declare_parameter(
            'state_topic', f'/{robot}/person_servo_node/state').value
        video_topic = self.declare_parameter('video_topic', f'/{robot}/eo/video').value
        out_topic = self.declare_parameter(
            'annotations_topic', f'/{robot}/eo/servo_annotations').value
        self.stamp_from_video = bool(self.declare_parameter('stamp_from_video', True).value)
        self.thickness = float(self.declare_parameter('thickness', 4.0).value)
        self.font_size = float(self.declare_parameter('font_size', 28.0).value)

        self.last_video_stamp = None
        self.was_engaged = None
        self.pub = self.create_publisher(ImageAnnotations, out_topic, 10)
        self.create_subscription(ServoState, state_topic, self.on_state, qos_profile_sensor_data)
        if self.stamp_from_video:
            self.create_subscription(CompressedVideo, video_topic, self.on_video,
                                     qos_profile_sensor_data)
        self.get_logger().info(
            f'servo overlay: {state_topic} -> {out_topic} '
            f'(stamps from {video_topic if self.stamp_from_video else "ServoState"})')

    def on_video(self, msg: CompressedVideo):
        self.last_video_stamp = msg.timestamp

    def on_state(self, msg: ServoState):
        out = ImageAnnotations()
        engaged = (msg.state in ENGAGED and msg.has_target
                   and msg.bbox_w > 0.0 and msg.bbox_h > 0.0)
        if engaged:
            stamp = (self.last_video_stamp if self.stamp_from_video and self.last_video_stamp
                     else msg.header.stamp)
            r, g, b = COLORS.get(msg.state, (0.1, 0.9, 0.2))
            x0, y0 = float(msg.bbox_x), float(msg.bbox_y)
            x1, y1 = x0 + float(msg.bbox_w), y0 + float(msg.bbox_h)
            box = PointsAnnotation()
            box.timestamp = stamp
            box.type = PointsAnnotation.LINE_LOOP
            box.points = [Point2(x=x0, y=y0), Point2(x=x1, y=y0),
                          Point2(x=x1, y=y1), Point2(x=x0, y=y1)]
            box.outline_color = Color(r=r, g=g, b=b, a=1.0)
            box.thickness = self.thickness
            label = TextAnnotation()
            label.timestamp = stamp
            label.position = Point2(x=x0, y=max(0.0, y0 - self.font_size - 4.0))
            label.text = (f'servo {STATE_NAMES.get(msg.state, msg.state)} '
                          f'{msg.detector_conf:.2f}')
            label.font_size = self.font_size
            label.text_color = Color(r=1.0, g=1.0, b=1.0, a=1.0)
            label.background_color = Color(r=0.0, g=0.0, b=0.0, a=0.6)
            out.points = [box]
            out.texts = [label]
        # Empty when not engaged: Foxglove replaces the previous annotations, so the
        # box clears as soon as the servo stops or loses the person.
        self.pub.publish(out)
        if engaged != self.was_engaged:
            self.get_logger().info(
                f'servo {STATE_NAMES.get(msg.state, msg.state)}: overlay '
                f'{"ON" if engaged else "off"}')
            self.was_engaged = engaged


def main():
    rclpy.init()
    node = ServoOverlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

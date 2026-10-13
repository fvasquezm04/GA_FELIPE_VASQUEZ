#!/usr/bin/env python3
"""
cinematica.py — Cinemática del robot Mecanum (Parte 4)

    ξ = [u, v, r]ᵀ                       velocidades en el marco del robot
    ω = [ω_di, ω_dd, ω_ti, ω_td]ᵀ        ruedas: delantera izq/der, trasera izq/der

    Directa : ξ = W ω     (W de 3x4, ver enunciado)
    Inversa : ω = W⁺ ξ

Lazo:
    /cmd_vel --W⁺--> /vel_rueda_* --> Gazebo (JointController) --> ruedas
    /joint_states --W--> ξ --η̇ = J(φ)ξ, Euler--> /odom + TF odom -> base_link

    ros2 run mi_mecanum_control cinematica --ros-args -p use_sim_time:=true
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster

# ===== TODO: parámetros físicos de SU variante (deben coincidir con el xacro) =====
A_RUEDA = 0.05     # a  : rueda_radio
LX = 0.3/2.0          # lx : distancia_ejes / 2
LY = 0.28/2.0          # ly : separacion_ruedas / 2

RUEDAS = ['del_izq', 'del_der', 'tra_izq', 'tra_der']
JOINTS = [f'joint_rueda_{p}' for p in RUEDAS]
TOPICOS = [f'/vel_rueda_{p}' for p in RUEDAS]


def matriz_W(a, lx, ly):
    """Cinemática directa del Mecanum: ξ = W ω (3x4)."""

    longitud = lx + ly

    return (a / 4.0) * np.array([
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, -1.0],
        [
            -1.0 / longitud,
            1.0 / longitud,
            -1.0 / longitud,
            1.0 / longitud,
        ],
    ], dtype=float)

def envolver(a):
    """Envuelve un ángulo a [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def integrar(x, y, phi, u, v, r, dt):
    """
    η̇ = J(φ) ξ e integración de Euler.

    u y v están expresadas en el marco móvil del robot.
    Devuelve (x, y, φ), con φ normalizado a [-π, π].
    """

    # Transformar las velocidades del marco del robot al marco odom.
    x_punto = math.cos(phi) * u - math.sin(phi) * v
    y_punto = math.sin(phi) * u + math.cos(phi) * v
    phi_punto = r

    # Integración de Euler.
    x_nuevo = x + x_punto * dt
    y_nuevo = y + y_punto * dt
    phi_nuevo = envolver(phi + phi_punto * dt)

    return x_nuevo, y_nuevo, phi_nuevo


class Cinematica(Node):
    def __init__(self):
        super().__init__('cinematica')
        self.W = matriz_W(A_RUEDA, LX, LY)
        self.W_pinv = np.linalg.pinv(self.W)   # TODO: ω = W⁺ ξ

        # Inversa
        self.create_subscription(Twist, '/cmd_vel', self.cb_cmd_vel, 10)
        self.pubs = [self.create_publisher(Float64, t, 10) for t in TOPICOS]

        # Directa + odometría
        self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)
        self.tf = TransformBroadcaster(self)

        self.x = self.y = self.phi = 0.0
        self.t_ant = None

        np.set_printoptions(precision=4, suppress=True)
        self.get_logger().info(f'\nW =\n{self.W}\nW⁺ =\n{self.W_pinv}')

    # ---------- Cinemática inversa: ω = W⁺ ξ ----------
    def cb_cmd_vel(self, msg):

        # ξ = [u, v, r]ᵀ
        xi = np.array([
            msg.linear.x,
            msg.linear.y,
            msg.angular.z,
        ], dtype=float)

        # ω = W⁺ ξ
        omega = self.W_pinv @ xi

        # Publicar cada velocidad angular en el orden:
        # del_izq, del_der, tra_izq, tra_der.
        for publicador, velocidad in zip(self.pubs, omega):
            mensaje = Float64()
            mensaje.data = float(velocidad)
            publicador.publish(mensaje)

    # ---------- Cinemática directa: ξ = W ω, y odometría ----------
    def cb_joints(self, msg):
        # ω en el orden de JOINTS
        try:
            omega = np.array([msg.velocity[msg.name.index(j)] for j in JOINTS])
        except (ValueError, IndexError):
            return

        ahora = self.get_clock().now()
        if self.t_ant is None:
            self.t_ant = ahora
            return
        dt = (ahora - self.t_ant).nanoseconds * 1e-9
        self.t_ant = ahora
        if dt <= 0.0:
            return

        # TODO: calcular u, v, r con W y actualizar self.x, self.y, self.phi
        u = v = r = 0.0

        # Cinemática directa:
        # ξ = W ω = [u, v, r]ᵀ
        xi = self.W @ omega

        u = float(xi[0])
        v = float(xi[1])
        r = float(xi[2])

        # Integración de la odometría mediante Euler.
        self.x, self.y, self.phi = integrar(
            self.x,
            self.y,
            self.phi,
            u,
            v,
            r,
            dt,
        )

        qz, qw = math.sin(self.phi / 2.0), math.cos(self.phi / 2.0)
        stamp = ahora.to_msg()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        # TODO: llenar odom.pose.pose (posición y orientación) y odom.twist.twist (u, v, r)

        # Pose del robot en el marco odom.
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        # Twist expresado en el marco base_link.
        odom.twist.twist.linear.x = u
        odom.twist.twist.linear.y = v
        odom.twist.twist.linear.z = 0.0

        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = r

        self.pub_odom.publish(odom)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf.sendTransform(t)


def main():
    rclpy.init()
    nodo = Cinematica()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

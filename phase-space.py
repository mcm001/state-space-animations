#!/usr/bin/env python

from manimlib.imports import *

# To watch one of these scenes, run the following:
# python -m manim example_scenes.py SquareToCircle -pl
#
# Use the flat -l for a faster rendering at a lower
# quality.
# Use -s to skip to the end and just save the final frame
# Use the -p to have the animation (or image, if -s was
# used) pop up once done.
# Use -n <number> to skip ahead to the n'th animation of a scene.
# Use -r <number> to specify a resolution (for example, -r 1080
# for a 1920x1080 video)

extra_accel = lambda point: 0.0  # use to simulate feedback/feedforward


class State:
    def __init__(self, position, velocity):
        self.theta = position
        self.omega = velocity


# noinspection PyAttributeOutsideInit
class PendulumCirclingOrigin(Scene):
    CONFIG = {
        "extra_accel_": lambda point: 0.0,
        "point_vector_max_len": 6.0,
        "show_state_point_vector": True,
        "hide_pendulum": False,
        "pendulum_config": {
            "initial_theta": 60 * DEGREES,
            "length": 2.0,
            "damping": 0,
            "top_point": ORIGIN,
        },
        "vector_field_config": {
            # "max_magnitude": 2,
            "delta_x": 0.5 * 1.5,
            "delta_y": 0.5 * 1.5,
            # "x_max": 6,
            "length_func": lambda norm: 0.6 * sigmoid(norm)
            # "color_by_arc_length": True,
            # "colors": [BLUE_E, GREEN, YELLOW, RED]
        },
        "coordinate_plane_config": {
            "x_max": 5 * PI / 2,
            "x_min": -5 * PI / 2,
            "y_max": 5,
            "y_min": -5
        }
    }

    def construct(self):
        global extra_accel
        extra_accel = self.extra_accel_
        self.state = State(self.pendulum_config["initial_theta"], 0.0)
        self.plane = NumberPlane(**self.coordinate_plane_config)
        self.create_pendulum_but_dont_add()
        self.create_vector_field()
        self.create_point_and_vec()
        self.add_pendulum()

        # self.wait(20)
        self.wait(5)

    def add_pendulum(self):
        self.add(self.pendulum)

    def get_evolving_trajectory(self, mobject, color=WHITE):
        trajectory = VMobject()
        trajectory.start_new_path(mobject.get_center())
        trajectory.set_stroke(color, opacity=1)

        def update_trajectory(traj):
            point = mobject.get_center()
            if get_norm(trajectory.points[-1] == point) > 0.05:
                traj.add_smooth_curve_to(point)

        trajectory.add_updater(update_trajectory)
        return trajectory

    def create_vector_field(self):
        plane = self.plane
        plane.add(plane.y_axis.get_labels())

        plane.x_axis.add_numbers(direction=DL)
        plane.add(plane.get_axis_labels("\\theta", "\\omega"))

        vector_field = self.vector_field = VectorField(self.pendulum_function, **self.vector_field_config)
        self.vector_field.sort(get_norm)
        self.add(plane, vector_field)

    def create_point_and_vec(self):
        pendulum: Pendulum = self.pendulum

        state_point = Dot().set_color(GREEN)
        state_point.add_updater(
            lambda point: state_point.move_to((np.array((self.pendulum.get_theta(), self.pendulum.get_omega(), 0.)))))

        def draw_vector_and_move_state_point():
            # Create a dot to represent our current state in state-space

            state_point_pos = state_point.get_center_of_mass()
            state_point_at_t = state_point_pos

            # Create a vector representing xdot at tour current point in state-space
            xdot_at_t = self.vector_field.func(state_point_at_t)
            multiple = np.clip(
                get_norm(xdot_at_t), -self.point_vector_max_len, self.point_vector_max_len
            )
            # vector = Vector(xdot_at_t / multiple)
            vector = Vector(xdot_at_t / multiple)
            vector.shift(state_point_pos)
            vector.set_color(GREEN)

            # return our point + vector mobj
            # vector.s(state_point_at_t)
            return vector

        self.state_point = state_point
        self.trajectory = self.get_evolving_trajectory(state_point)
        if (self.show_state_point_vector):
            state_vector = always_redraw(draw_vector_and_move_state_point)
            self.add(state_vector)
        self.add(self.trajectory, self.state_point)

    def pendulum_function(self, point):
        x, y = self.plane.point_to_coords(point)
        return pendulum_vector_field_func(np.array((x, y, 0.)), L=self.pendulum_config['length'])

    def create_pendulum_but_dont_add(self):
        pendulum = self.pendulum = Pendulum(**self.pendulum_config)
        pendulum.add_theta_label()
        pendulum.add_velocity_vector()
        pendulum.start_swinging()

        pendulum = self.pendulum
        background_rectangle = Rectangle(height=6, width=6, opacity=1.0, color=GREEN) \
            .set_fill(color=BLACK, opacity=1.0) \
            .shift(DOWN * 0.5)
        pendulum.add_to_back(background_rectangle)
        pendulum.scale_in_place(0.5)

        if (self.hide_pendulum is False):
            pendulum.move_to(TOP + LEFT_SIDE + (RIGHT + DOWN) * 0.25, aligned_edge=pendulum.get_corner(UP + LEFT))
        else:
            pendulum.move_to((TOP + LEFT_SIDE) * 1.1, aligned_edge=pendulum.get_corner(DOWN + RIGHT))


def pendulum_vector_field_func(point, L=3, g=9.8):
    x, y = point[:2]
    x_dot = np.array([[0, 1], [-g / L, 0]]) @ (np.array([[math.sin(x)], [y]]))
    a_dot_x = np.array([x_dot[0, 0], x_dot[1, 0], 0.0])

    # x, y = point[:2]
    extra_acceleration = extra_accel(np.array((x, y)))
    # a_dot_x = np.array([
    #     y,
    #     -np.sqrt(g / L) * np.sin(x) - mu * y,
    #     0.,
    # ])
    # print("normal: %s, extra: %s" % (a_dot_x, extra_acceleration))
    return a_dot_x + extra_acceleration
    # return np.array([1, 0, 0])


class Pendulum(VGroup):
    CONFIG = {
        "length": 2,
        "weight_diameter": 0.5,
        "initial_theta": 0.3,
        "omega": 0,
        "damping": 0.1,
        "top_point": 2 * UP,
        "rod_style": {
            "stroke_width": 3,
            "stroke_color": LIGHT_GREY,
            "sheen_direction": UP,
            "sheen_factor": 1,
        },
        "weight_style": {
            "stroke_width": 0,
            "fill_opacity": 1,
            "fill_color": GREY_BROWN,
            "sheen_direction": UL,
            "sheen_factor": 0.5,
            "background_stroke_color": BLACK,
            "background_stroke_width": 3,
            "background_stroke_opacity": 0.5,
        },
        "dashed_line_config": {
            "num_dashes": 25,
            "stroke_color": WHITE,
            "stroke_width": 2,
        },
        "angle_arc_config": {
            "radius": 1,
            "stroke_color": WHITE,
            "stroke_width": 2,
        },
        "velocity_vector_config": {
            "color": RED,
        },
        "theta_label_height": 0.25,
        "set_theta_label_height_cap": False,
        "n_steps_per_frame": 100,
        "include_theta_label": True,
        "include_velocity_vector": False,
        "velocity_vector_multiple": 0.5,
        "max_velocity_vector_length_to_length_ratio": 0.5,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_fixed_point()
        self.create_rod()
        self.create_weight()
        self.rotating_group = VGroup(self.rod, self.weight)
        self.create_dashed_line()
        self.create_angle_arc()
        if self.include_theta_label:
            self.add_theta_label()
        if self.include_velocity_vector:
            self.add_velocity_vector()

        self.set_theta(self.initial_theta)
        self.update()

    def create_fixed_point(self):
        self.fixed_point_tracker = VectorizedPoint(self.top_point)
        self.add(self.fixed_point_tracker)
        return self

    def create_rod(self):
        rod = self.rod = Line(UP, DOWN)
        rod.set_height(self.length)
        rod.set_style(**self.rod_style)
        rod.move_to(self.get_fixed_point(), UP)
        self.add(rod)

    def create_weight(self):
        weight = self.weight = Circle()
        weight.set_width(self.weight_diameter)
        weight.set_style(**self.weight_style)
        weight.move_to(self.rod.get_end())
        self.add(weight)

    def create_dashed_line(self):
        line = self.dashed_line = DashedLine(
            self.get_fixed_point(),
            self.get_fixed_point() + self.length * DOWN,
            **self.dashed_line_config
        )
        line.add_updater(
            lambda l: l.move_to(self.get_fixed_point(), UP)
        )
        self.add_to_back(line)

    def create_angle_arc(self):
        self.angle_arc = always_redraw(lambda: Arc(
            arc_center=self.get_fixed_point(),
            start_angle=-90 * DEGREES,
            angle=self.get_arc_angle_theta(),
            **self.angle_arc_config,
        ))
        self.add(self.angle_arc)

    def get_arc_angle_theta(self):
        # Might be changed in certain scenes
        return self.get_theta()

    def add_velocity_vector(self):
        def make_vector():
            omega = self.get_omega()
            theta = self.get_theta()
            mvlr = self.max_velocity_vector_length_to_length_ratio
            max_len = mvlr * self.rod.get_length()
            vvm = self.velocity_vector_multiple
            multiple = np.clip(
                vvm * omega, -max_len, max_len
            )
            vector = Vector(
                multiple * RIGHT,
                **self.velocity_vector_config,
            )
            vector.rotate(theta, about_point=ORIGIN)
            vector.shift(self.rod.get_end())
            return vector

        self.velocity_vector = always_redraw(make_vector)
        self.add(self.velocity_vector)
        return self

    def add_theta_label(self):
        self.theta_label = always_redraw(self.get_label)
        self.add(self.theta_label)

    def get_label(self):
        label = TexMobject("\\theta")
        label.set_height(self.theta_label_height)
        if self.set_theta_label_height_cap:
            max_height = self.angle_arc.get_width()
            if label.get_height() > max_height:
                label.set_height(max_height)
        top = self.get_fixed_point()
        arc_center = self.angle_arc.point_from_proportion(0.5)
        vect = arc_center - top
        norm = get_norm(vect)
        vect = normalize(vect) * (norm + self.theta_label_height)
        label.move_to(top + vect)
        return label

    #
    def get_theta(self):
        theta = self.rod.get_angle() - self.dashed_line.get_angle()
        theta = (theta + PI) % TAU - PI
        return theta

    def get_unbounded_theta(self):
        return self.rod.get_angle() - self.dashed_line.get_angle()

    def set_theta(self, theta):
        self.rotating_group.rotate(
            theta - self.get_theta()
        )
        self.rotating_group.shift(
            self.get_fixed_point() - self.rod.get_start(),
        )
        return self

    def get_omega(self):
        return self.omega

    def set_omega(self, omega):
        self.omega = omega
        return self

    def get_fixed_point(self):
        return self.fixed_point_tracker.get_location()

    #
    def start_swinging(self):
        self.add_updater(Pendulum.update_by_gravity)

    def end_swinging(self):
        self.remove_updater(Pendulum.update_by_gravity)

    def update_by_gravity(self, dt):

        theta = self.get_theta()
        omega = self.get_omega()
        if (theta > 3):
            ohno = 4

        nspf = self.n_steps_per_frame
        for x in range(nspf):
            d_theta = omega * dt / nspf
            d_omega = pendulum_vector_field_func(np.array((theta, omega, 0.)), L=self.length)[1] * dt / nspf

            theta += d_theta
            omega += d_omega

        self.set_theta(theta)
        self.set_omega(omega)
        return self


class UnstableFeedForwardAtHorizontal(PendulumCirclingOrigin):
    CONFIG = {
        "extra_accel_": lambda point: np.array((0.0, 4.9, 0.0)),
        "pendulum_config": {
            "initial_theta": 60 * DEGREES,
        },
    }


class FeedbackWithArmAtHorizontal(PendulumCirclingOrigin):
    CONFIG = {
        "extra_accel_": lambda point: (
                np.array((0.0, 4.9, 0.0)) + np.array((1.0 * (PI / 2.0 - point[0]), 1.0 * (0.0 - point[1]), 0.0))),
        "pendulum_config": {
            "initial_theta": 30 * DEGREES,
        },
        "show_state_point_vector": False,
        "hide_pendulum": True
    }

    def construct(self):
        global extra_accel
        extra_accel = self.extra_accel_
        self.plane = NumberPlane(**self.coordinate_plane_config)
        self.create_pendulum_but_dont_add()
        self.create_vector_field()
        self.create_point_and_vec()
        self.add_pendulum()

        # self.wait(20)
        self.wait(4)


class DotWithState(Dot):
    def __init__(self, n_steps_per_frame, length, gravity, plane: NumberPlane, point=ORIGIN, state=None, **kwargs):
        Dot.__init__(self, point=point, **kwargs)
        if state is None:
            self.state = State(point[0], point[1])
        else:
            self.state = state
        self.n_steps_per_frame = n_steps_per_frame
        self.length = length
        self.gravity = gravity
        self.plane: NumberPlane = plane

    def get_state(self):
        return self.state

    def get_theta(self):
        return self.state.theta

    def get_omega(self):
        return self.state.omega

    def set_theta(self, theta):
        self.state.theta = theta

    def set_omega(self, omega):
        self.state.omega = omega

    def update_position(self):
        x, y = self.plane.point_to_coords(np.array((self.state.theta, self.state.omega, 0.0)))
        self.move_to([x, y, 0])

    def start_swinging(self):
        self.add_updater(DotWithState.update_state_by_gravity)

    def update_state_by_gravity(self, dt):

        theta = self.get_theta()
        omega = self.get_omega()

        nspf = self.n_steps_per_frame
        for x in range(nspf):
            d_theta = omega * dt / nspf
            d_omega = pendulum_vector_field_func(np.array((theta, omega, 0.)), L=self.length)[1] * dt / nspf

            theta += d_theta
            omega += d_omega

        self.set_theta(theta)
        self.set_omega(omega)
        self.update_position()
        return self


class ShowMultipleFeedback(PendulumCirclingOrigin):
    CONFIG = {
        "gravity": 9.8,
        "extra_accel_": lambda point: (
                np.array((0.0, 0.0, 0.0)) + 0 * np.array(((0 - point[0]), (0 - point[1]), 0.0))),
        "show_state_point_vector": False,
        "hide_pendulum": True,
        "n_steps_per_frame": 100
    }

    def construct(self):
        global extra_accel
        extra_accel = self.extra_accel_
        self.plane = NumberPlane(**self.coordinate_plane_config)
        self.create_vector_field()
        self.create_and_add_points()

        self.wait(10.0)

    def create_and_add_points(self):
        # create 10 points between -5 and 5 on x and -3 and 3 on y

        dot = DotWithState(self.n_steps_per_frame, self.pendulum_config["length"], self.gravity, self.plane,
                           point=np.array((2, 1, 0)))
        trajectory = (self.get_evolving_trajectory(dot, color=WHITE))
        dot.start_swinging()
        self.add(trajectory, dot)

        for x in np.arange(-7, 7.1, 2.0):
            for y in np.arange(-4, 4.1, 2.0):
                dot = DotWithState(self.n_steps_per_frame, self.pendulum_config["length"], self.gravity, self.plane,
                                   point=np.array((x, y, 0)))
                trajectory = self.get_evolving_trajectory(dot, WHITE)
                dot.start_swinging()
                self.add(trajectory, dot)

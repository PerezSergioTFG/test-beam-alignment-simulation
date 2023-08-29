import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click


def q_realtrack(n, q_max, distribution, random_seed=None):
    """
    Returns an array of n random values following either
        a gaussian distribution with mean = 0, sigma = q_max/4.
        or a uniform distribution in the range (-q_max,+q_max)
    Each value represents the real track position in the coordinate q (q_realtrack) corresponding to a single track.
    The real track position is defined as the position at which the real track intersects the DUT plane.
    """
    np.random.seed(random_seed)

    if distribution == "gaussian":
        # We chose standard_deviation = x_max/4 so that x>x_max has a probability of four sigma
        q_realtrack = np.random.normal(loc=0, scale=q_max / 4, size=n)
    elif distribution == "uniform":
        q_realtrack = np.random.uniform(low=-q_max, high=q_max, size=n)
    else:
        raise ValueError(
            f' "{distribution}" is not a valid value of distribution, use "gaussian" or "uniform"'
        )

    return q_realtrack


def q_closest_pixel_center(q_realtrack):
    q = q_realtrack.copy().astype(float)
    q_measured = np.piecewise(
        q,
        [q < 0, q >= 0],
        [lambda q: q.astype(int) - 0.5, lambda q: q.astype(int) + 0.5],
    )
    return q_measured


def q_measured_mcs4(q_realtrack):
    R = 0.25
    q_pixel_center = q_closest_pixel_center(q_realtrack)
    q_range_01 = q_realtrack - (q_pixel_center - 0.5)
    x = q_range_01  # simply so that the function is more readable
    q_measured = np.piecewise(
        x,
        [
            x < 0.25,
            (x >= 0.25) & (x < 0.75),
            x >= 0.75,
        ],
        [
            lambda x: (
                R**2 * np.arccos(x / R) - x * np.sqrt(2 * R * (R + x) - (R + x) ** 2)
            )
            * (-16 / np.pi)
            + 0.5,
            0.5,
            lambda x: (
                R**2 * np.arccos((x - 1) / R)
                - (x - 1) * np.sqrt(2 * R * (R + (x - 1)) - (R + (x - 1)) ** 2)
            )
            * (-16 / np.pi)
            + 1.5,
        ],
    )
    q_measured = (q_pixel_center - 0.5) + q_measured
    return q_measured


def cluster_size_mcs4(x_realtrack, y_realtrack):
    x_range_01 = x_realtrack - (q_closest_pixel_center(x_realtrack) - 0.5)
    y_range_01 = y_realtrack - (q_closest_pixel_center(y_realtrack) - 0.5)

    x, y = x_range_01, y_range_01  # simply for readability of the functions bellow

    x_left = np.where(x < 0.25, 1, 0)
    x_center = np.where((x >= 0.25) & (x < 0.75), 1, 0)
    x_right = np.where(x >= 0.75, 1, 0)

    y_bottom = np.where(y < 0.25, 1, 0)
    y_middle = np.where((y >= 0.25) & (y < 0.75), 1, 0)
    y_top = np.where(y >= 0.75, 1, 0)

    bottom_left = x_left * y_bottom
    bottom_center = x_center * y_bottom
    bottom_right = x_right * y_bottom

    middle_left = x_left * y_middle
    middle_center = x_center * y_middle
    middle_right = x_right * y_middle

    top_left = x_left * y_top
    top_center = x_center * y_top
    top_right = x_right * y_top

    cluster_size = (
        4 * (bottom_left + bottom_right + top_left + top_right)
        + 2 * (bottom_center + middle_left + middle_right + top_center)
        + 1 * (middle_center)
    )
    return cluster_size


def xy_measured(x_realtrack, y_realtrack, maximum_cluster_size):
    if maximum_cluster_size == 1:
        x_measured = q_closest_pixel_center(x_realtrack)
        y_measured = q_closest_pixel_center(y_realtrack)
        cluster_size = np.ones(x_realtrack.size).astype(int)
    elif maximum_cluster_size == 4:
        x_measured = q_measured_mcs4(x_realtrack)
        y_measured = q_measured_mcs4(y_realtrack)
        cluster_size = cluster_size_mcs4(x_realtrack, y_realtrack).astype(int)
    else:
        raise ValueError(
            f' "{maximum_cluster_size}" is not a valid value of maximum_cluster_size, use 1 or 4'
        )
    return x_measured, y_measured, cluster_size


def q_trackfit(
    x_realtrack,
    y_realtrack,
    thetax_realtrack,
    thetay_realtrack,
    x_misaligned,
    y_misaligned,
    z_misaligned,
    alpha,
    beta,
    gamma,
):
    # We have to convert al the angles to radians because numpy trigonometric functions interpret input angles as radians
    thetax_realtrack = np.radians(thetax_realtrack)
    thetay_realtrack = np.radians(thetay_realtrack)
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    x_trackfit = (
        (np.cos(beta) * np.cos(gamma)) * x_realtrack
        + (np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma))
        * y_realtrack
        + x_misaligned
        - z_misaligned * np.tan(thetax_realtrack)
        # (-) explanation: Due to the criteria that we have chosen to define thetaq_realtrack and z_misaligned,
        # z_misaligned >0, thetax_realtrack>0 => Delta_x<0
        # In practice, this sign has no effect on the calculations because the distribution of thetax_realtrack is symmetrical
        # for positive and negative values, and is independent of any other variable.
    )
    y_trackfit = (
        (np.cos(beta) * np.sin(gamma)) * x_realtrack
        + (np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma))
        * y_realtrack
        + y_misaligned
        - z_misaligned * np.tan(thetay_realtrack)
    )
    return x_trackfit, y_trackfit


def generate_hits_positions(
    n,
    beam_positions_distribution,
    incidence_angles_distribution,
    random_seed,
    x_max,
    y_max,
    thetax_max,
    thetay_max,
    maximum_cluster_size,
    x_misaligned,
    y_misaligned,
    z_misaligned,
    alpha,
    beta,
    gamma,
):
    """
    Arguments:
        n: INT Number of data points
        x_misaligned: FLOAT Translation in x (units = DUTpixels) (alignment parameter)
        y_misaligned: FLOAT Translation in y (units = DUTpixels) (alignment parameter)
        alpha: FLOAT Rotation around x (units = degrees) (alignment parameter)
        beta: FLOAT Rotation around y (units = degrees) (alignment parameter)
        gamma: FLOAT Rotation around z (units = degrees) (alignment parameter)

    Returns:
        x_realtrack: NP.ARRAY (size = n) x position where the track actually intercepts the DUT plane (units = pixels)
        y_realtrack: NP.ARRAY (size = n) y position where the track actually intercepts the DUT plane (units = pixels)
        x_measured: NP.ARRAY (size = n) x position measured by the DUT (units = pixels)
        y_measured: NP.ARRAY (size = n) y position measured by the DUT (units = pixels)
        x_trackfit: NP.ARRAY (size = n) x position reconstructed by the telescope (units = pixels)
        y_trackfit: NP.ARRAY (size = n) y position reconstructed by the telescope (units = pixels)
        Delta_x: NP.ARRAY (size = n) = x_trackfit- x_measured
        Delta_y: NP.ARRAY (size = n) = y_trackfit- y_measured

    """

    ## Real Track Position
    x_realtrack = q_realtrack(n, x_max, beam_positions_distribution, random_seed)
    y_realtrack = q_realtrack(n, y_max, beam_positions_distribution, random_seed + 1)
    # If we used (random_seed) instead of (random_seed+1), then x_realtrack == y_realtrack and the resulting scattering plots would be a line
    thetax_realtrack = q_realtrack(
        n, thetax_max, incidence_angles_distribution, random_seed + 2
    )
    thetay_realtrack = q_realtrack(
        n, thetay_max, incidence_angles_distribution, random_seed + 3
    )

    # Position measured by DUT
    x_measured, y_measured, cluster_size = xy_measured(
        x_realtrack, y_realtrack, maximum_cluster_size
    )

    # Position reconstructed by the telescope
    x_trackfit, y_trackfit = q_trackfit(
        x_realtrack,
        y_realtrack,
        thetax_realtrack,
        thetay_realtrack,
        x_misaligned,
        y_misaligned,
        z_misaligned,
        alpha,
        beta,
        gamma,
    )

    # Residuals
    Delta_x = x_trackfit - x_measured
    Delta_y = y_trackfit - y_measured

    return (
        x_realtrack,
        y_realtrack,
        thetax_realtrack,
        thetay_realtrack,
        x_measured,
        y_measured,
        x_trackfit,
        y_trackfit,
        Delta_x,
        Delta_y,
        cluster_size,
    )


def hits_scatter_plots(
    x_realtrack, y_realtrack, x_measured, y_measured, x_trackfit, y_trackfit
):
    """
    Plots hits positions generated by the function generate_hits_positions() in 3 subplots.
    """
    figure, axis = plt.subplots(1, 3, figsize=(12, 3.5))

    """ DISPLAY BEAM SIZE +2
    square_side = max(
        abs(np.min(x_trackfit).astype(int)),
        np.max(x_trackfit).astype(int),
        abs(np.min(y_trackfit).astype(int)),
        np.max(y_trackfit).astype(int),
    )  # We want the scatter plots to be squares where all hits are visible.

    x_min_displayed = -square_side - 2  # minimum value of x we want to display
    x_max_displayed = +square_side + 2  # maximum value of x we want to display
    y_min_displayed = -square_side - 2  # minimum value of y we want to display
    y_max_displayed = +square_side + 2  # maximum value of y we want to display
    """

    dut_size = 16

    x_min_displayed, x_max_displayed, y_min_displayed, y_max_displayed = (
        -dut_size,
        +dut_size,
        -dut_size,
        +dut_size,
    )

    # Plot: (x_trackfit, y_trackfit)
    axis[0].scatter(x=x_trackfit, y=y_trackfit, s=3, color="darkorange")
    axis[0].set_title("Telescope")
    axis[0].set_xlim(x_min_displayed, x_max_displayed)
    axis[0].set_ylim(y_min_displayed, y_max_displayed)
    axis[0].set_xticks(np.arange(x_min_displayed, x_max_displayed))
    axis[0].set_yticks(np.arange(y_min_displayed, y_max_displayed))
    axis[0].tick_params(labelsize=0)
    axis[0].grid(color="darkgray", linewidth=1)

    # Plot: (x_realtrack, y_realtrack)
    axis[1].scatter(x=x_realtrack, y=y_realtrack, s=3, color="k")
    axis[1].set_title("True")
    axis[1].set_xlim(x_min_displayed, x_max_displayed)
    axis[1].set_ylim(y_min_displayed, y_max_displayed)
    axis[1].set_xticks(np.arange(x_min_displayed, x_max_displayed))
    axis[1].set_yticks(np.arange(y_min_displayed, y_max_displayed))
    axis[1].tick_params(labelsize=0)
    # There's probably a better solution to display only some ticks
    axis[1].grid(color="darkgray", linewidth=1)

    # Plot: (x_measured, y_measured)
    axis[2].scatter(x=x_measured, y=y_measured, s=3, color="blue")
    axis[2].set_title("DUT")
    axis[2].set_xlim(x_min_displayed, x_max_displayed)
    axis[2].set_ylim(y_min_displayed, y_max_displayed)
    axis[2].set_xticks(np.arange(x_min_displayed, x_max_displayed))
    axis[2].set_yticks(np.arange(y_min_displayed, y_max_displayed))
    axis[2].tick_params(labelsize=0)
    axis[2].grid(color="darkgray", linewidth=1)

    plt.subplots_adjust(
        top=0.88, bottom=0.1, left=0.125, right=0.9, hspace=0.2, wspace=0.2
    )

    plt.show()


def residual_plots(
    x_trackfit, y_trackfit, thetax_realtrack, thetay_realtrack, Delta_x, Delta_y
):
    """
    PLots residuals generated by the function generate_hits_positions() in 6 subplots.
    """
    # ALL PLOTS
    # figure, axis = plt.subplots(2, 4, figsize=(12, 5))

    # ONE PLOT
    figure, axis = plt.subplots(2, 4, figsize=(13, 6))

    figure.tight_layout(h_pad=4, w_pad=3)

    # First we obtain the maximum and minimum values that we want to display for each variable
    x_min_displayed = (
        np.min(x_trackfit).astype(int) - 1
    )  # minimum value of x we want to display
    x_max_displayed = (
        np.max(x_trackfit).astype(int) + 1
    )  # maximum value of x we want to display
    y_min_displayed = (
        np.min(y_trackfit).astype(int) - 1
    )  # minimum value of y we want to display
    y_max_displayed = (
        np.max(y_trackfit).astype(int) + 1
    )  # maximum value of y we want to display
    Deltax_min_displayed = (
        np.min(Delta_x).astype(int) - 1
    )  # minimum value of Delta_x we want to display
    Deltax_max_displayed = (
        np.max(Delta_x).astype(int) + 1
    )  # maximum value of Delta_x we want to display
    Deltay_min_displayed = (
        np.min(Delta_y).astype(int) - 1
    )  # minimum value of Delta_y we want to display
    Deltay_max_displayed = (
        np.max(Delta_y).astype(int) + 1
    )  # maximum value of Delta_y we want to display

    # Delta_x distribution
    sns.set_style("whitegrid")
    sns.histplot(ax=axis[0, 0], x=Delta_x, kde=True)
    # print(np.std(Delta_x))
    axis[0, 0].set_xlim(Deltax_min_displayed, Deltax_max_displayed)
    axis[0, 0].set_xlabel(r"$\Delta x$ (pixels)")
    axis[0, 0].set_ylabel(r"counts")
    axis[0, 0].set_title(r"$\Delta x$ histogram")

    # Delta_y distribution
    sns.histplot(ax=axis[1, 0], x=Delta_y, kde=True)
    # print(np.std(Delta_y))
    axis[1, 0].set_xlim(Deltay_min_displayed, Deltay_max_displayed)
    axis[1, 0].set_xlabel(r"$\Delta y$ (pixels)")
    axis[1, 0].set_ylabel(r"counts")
    axis[1, 0].set_title(r"$\Delta y$ histogram")

    # Delta_x vs y
    sns.scatterplot(ax=axis[0, 1], x=y_trackfit, y=Delta_x, s=2)
    # Not sure if x input is correct
    axis[0, 1].set_xlim(y_min_displayed, y_max_displayed)
    axis[0, 1].set_xlabel(r"$y_{telescope}$ (pixels)")
    axis[0, 1].set_ylabel(r"$\Delta x$ (pixels)")
    axis[0, 1].set_title(r"$\Delta x$ vs. $y_{telescope}$")

    # Delta_y vs x
    sns.scatterplot(ax=axis[1, 1], x=x_trackfit, y=Delta_y, s=2)
    axis[1, 1].set_xlim(x_min_displayed, x_max_displayed)
    axis[1, 1].set_xlabel(r"$x_{telescope}$ (pixels)")
    axis[1, 1].set_ylabel(r"$\Delta y$ (pixels)")
    axis[1, 1].set_title(r"$\Delta y$ vs. $x_{telescope}$")

    # Delta_x vs x
    sns.scatterplot(ax=axis[0, 2], x=x_trackfit, y=Delta_x, s=2)
    # Not sure if x input is correct
    axis[0, 2].set_xlim(x_min_displayed, x_max_displayed)
    axis[0, 2].set_xlabel(r"$x_{telescope}$ (pixels)")
    axis[0, 2].set_ylabel(r"$\Delta x$ (pixels)")
    axis[0, 2].set_title(r"$\Delta x$ vs. $x_{telescope}$")

    # Delta_y vs x
    sns.scatterplot(ax=axis[1, 2], x=y_trackfit, y=Delta_y, s=2)
    axis[1, 2].set_xlim(y_min_displayed, y_max_displayed)
    axis[1, 2].set_xlabel(r"$y_{telescope}$ (pixels)")
    axis[1, 2].set_ylabel(r"$\Delta y$ (pixels)")
    axis[1, 2].set_title(r"$\Delta y$ vs. $y_{telescope}$")

    # Delta_x vs thetax
    sns.scatterplot(ax=axis[0, 3], x=thetax_realtrack, y=Delta_x, s=2)
    axis[0, 3].set_xlim(np.min(thetax_realtrack), np.max(thetax_realtrack))
    axis[0, 3].set_xlabel(r"$\theta_x$ (º)")
    axis[0, 3].set_ylabel(r"$\Delta x$ (pixels)")
    axis[0, 3].set_title(r"$\Delta x$ vs. $\theta_x$")

    # Delta_y vs x
    sns.scatterplot(ax=axis[1, 3], x=thetay_realtrack, y=Delta_y, s=2)
    axis[1, 3].set_xlim(np.min(thetay_realtrack), np.max(thetay_realtrack))
    axis[1, 3].set_xlabel(r"$ \theta_y $ (º)")
    axis[1, 3].set_ylabel(r"$\Delta y$ (pixels)")
    axis[1, 3].set_title(r"$\Delta y$ vs. $\theta_y$")

    """ALL PLOTS
    plt.subplots_adjust(
        top=0.954, bottom=0.092, left=0.039, right=0.975, hspace=0.476, wspace=0.416
    )
    """

    plt.subplots_adjust(
        top=0.954, bottom=0.092, left=0.050, right=0.975, hspace=0.476, wspace=0.416
    )

    plt.show()


def observables_plots(
    x_realtrack,
    y_realtrack,
    thetax_realtrack,
    thetay_realtrack,
    x_measured,
    y_measured,
    x_trackfit,
    y_trackfit,
    Delta_x,
    Delta_y,
    cluster_size,
):
    """
    PLots residuals generated by the function generate_hits_positions() in 6 subplots.
    """
    figure, axis = plt.subplots(2, 3, figsize=(12, 5))
    figure.tight_layout(h_pad=4, w_pad=3)

    # First we obtain the maximum and minimum values that we want to display for each variable
    x_min_displayed = (
        np.min(x_trackfit).astype(int) - 1
    )  # minimum value of x we want to display
    x_max_displayed = (
        np.max(x_trackfit).astype(int) + 1
    )  # maximum value of x we want to display
    y_min_displayed = (
        np.min(y_trackfit).astype(int) - 1
    )  # minimum value of y we want to display
    y_max_displayed = (
        np.max(y_trackfit).astype(int) + 1
    )  # maximum value of y we want to display
    Deltax_min_displayed = (
        np.min(Delta_x).astype(int) - 1
    )  # minimum value of Delta_x we want to display
    Deltax_max_displayed = (
        np.max(Delta_x).astype(int) + 1
    )  # maximum value of Delta_x we want to display
    Deltay_min_displayed = (
        np.min(Delta_y).astype(int) - 1
    )  # minimum value of Delta_y we want to display
    Deltay_max_displayed = (
        np.max(Delta_y).astype(int) + 1
    )  # maximum value of Delta_y we want to display

    sns.set_style("whitegrid")

    ## PLOTS
    # cluster_size vs x_realtrack and y-realtrack
    sns.scatterplot(
        ax=axis[0, 0],
        x=x_realtrack,
        y=y_realtrack,
        hue=cluster_size,
        s=6,
        palette="coolwarm",
    )
    axis[0, 0].set_xlim(x_min_displayed, x_max_displayed)
    axis[0, 0].set_ylim(y_min_displayed, y_max_displayed)
    axis[0, 0].set_xlabel(r"x_realtrack (pixels)")
    axis[0, 0].set_ylabel(r"y_realtrack (pixels)")
    axis[0, 0].set_title(r"cluster_size vs. x_realtrack and y_reatrack")

    # cluster_size vs x_realtrack and y-realtrack
    sns.scatterplot(
        ax=axis[0, 1],
        x=x_measured,
        y=y_measured,
        hue=cluster_size,
        s=6,
        palette="coolwarm",
    )
    axis[0, 1].set_xlim(x_min_displayed, x_max_displayed)
    axis[0, 1].set_ylim(y_min_displayed, y_max_displayed)
    axis[0, 1].set_xlabel(r"x_measured (pixels)")
    axis[0, 1].set_ylabel(r"y_measured (pixels)")
    axis[0, 1].set_title(r"cluster_size vs. x_measured and y_measured")

    # cluster_size vs x_realtrack and y-realtrack
    sns.scatterplot(
        ax=axis[0, 2],
        x=x_trackfit,
        y=y_trackfit,
        hue=cluster_size,
        s=6,
        palette="coolwarm",
    )
    axis[0, 2].set_xlim(x_min_displayed, x_max_displayed)
    axis[0, 2].set_ylim(y_min_displayed, y_max_displayed)
    axis[0, 2].set_xlabel(r"x_trackfit (pixels)")
    axis[0, 2].set_ylabel(r"y_trackfit (pixels)")
    axis[0, 2].set_title(r"cluster_size vs. x_trackfit and y_trackfit")

    # cluster_size vs x_realtrack and y-realtrack
    sns.scatterplot(
        ax=axis[1, 0],
        x=x_realtrack,
        y=y_realtrack,
        hue=cluster_size,
        s=6,
        palette="coolwarm",
    )
    axis[1, 0].set_xlim(x_min_displayed, x_max_displayed)
    axis[1, 0].set_ylim(y_min_displayed, y_max_displayed)
    axis[1, 0].set_xlabel(r"x_realtrack (pixels)")
    axis[1, 0].set_ylabel(r"y_realtrack (pixels)")
    axis[1, 0].set_title(r"cluster_size vs. x_realtrack and y_reatrack")

    # cluster_size vs x_realtrack and y-realtrack
    sns.scatterplot(
        ax=axis[1, 1],
        x=x_measured,
        y=y_measured,
        hue=cluster_size,
        s=6,
        palette="coolwarm",
    )
    axis[1, 1].set_xlim(x_min_displayed, x_max_displayed)
    axis[1, 1].set_ylim(y_min_displayed, y_max_displayed)
    axis[1, 1].set_xlabel(r"x_measured (pixels)")
    axis[1, 1].set_ylabel(r"y_measured (pixels)")
    axis[1, 1].set_title(r"cluster_size vs. x_measured and y_measured")

    # cluster_size vs x_realtrack and y-realtrack
    sns.scatterplot(
        ax=axis[1, 2],
        x=x_trackfit,
        y=y_trackfit,
        hue=cluster_size,
        s=6,
        palette="coolwarm",
    )
    axis[1, 2].set_xlim(x_min_displayed, x_max_displayed)
    axis[1, 2].set_ylim(y_min_displayed, y_max_displayed)
    axis[1, 2].set_xlabel(r"x_trackfit (pixels)")
    axis[1, 2].set_ylabel(r"y_trackfit (pixels)")
    axis[1, 2].set_title(r"cluster_size vs. x_trackfit and y_trackfit")
    """
    plt.subplots_adjust(
        top=0.954, bottom=0.092, left=0.039, right=0.975, hspace=0.476, wspace=0.416
    )
    """

    plt.show()


def simulate(
    n_sim,
    x_misaligned,
    y_misaligned,
    z_misaligned,
    alpha,
    beta,
    gamma,
    beam_positions_distribution,
    incidence_angles_distribution,
    maximum_cluster_size,
):
    """
    Uses the functions generate_hits_positions(), hits_scatter_plots(), and residual_plots()
    to perform simple MC toy to evaluate the DUT alignment.
    """
    (
        x_realtrack,
        y_realtrack,
        thetax_realtrack,
        thetay_realtrack,
        x_measured,
        y_measured,
        x_trackfit,
        y_trackfit,
        Delta_x,
        Delta_y,
        cluster_size,
    ) = generate_hits_positions(
        n=n_sim,
        maximum_cluster_size=maximum_cluster_size,  # use 1 or 4
        x_misaligned=x_misaligned,
        y_misaligned=y_misaligned,
        z_misaligned=z_misaligned,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        x_max=10,
        y_max=10,
        thetax_max=1,
        thetay_max=1,
        beam_positions_distribution=beam_positions_distribution,  # use "gaussian" or "uniform"
        incidence_angles_distribution=incidence_angles_distribution,  # use "gaussian" or "uniform"
        random_seed=1,
    )

    hits_scatter_plots(
        x_realtrack, y_realtrack, x_measured, y_measured, x_trackfit, y_trackfit
    )

    residual_plots(
        x_trackfit, y_trackfit, thetax_realtrack, thetay_realtrack, Delta_x, Delta_y
    )


def get_model_parameters(model):
    if model == "toy_model":
        return "uniform", "uniform", 1
    if model == "gaussian":
        return "gaussian", "gaussian", 1
    if model == "charge_sharing":
        return "gaussian", "gaussian", 1


@click.command()
@click.option(
    "-n", "--n-sim", default=1000, type=click.INT, help="Number of data points"
)
@click.option(
    "-x",
    "--x-misaligned",
    type=click.FLOAT,
    default=0,
    help="Translation in x (units = DUTpixels) (alignment parameter)",
)
@click.option(
    "-y",
    "--y-misaligned",
    type=click.FLOAT,
    default=0,
    help="Translation in y (units = DUTpixels) (alignment parameter)",
)
@click.option(
    "-z",
    "--z-misaligned",
    type=click.FLOAT,
    default=0,
    help="Translation in z (units = DUTpixels) (alignment parameter)",
)
@click.option(
    "-a",
    "--alpha",
    type=click.FloatRange(min=0, max=360),
    default=0,
    help="Rotation around x (units = degrees) (alignment parameter)",
)
@click.option(
    "-b",
    "--beta",
    type=click.FloatRange(min=0, max=360),
    default=0,
    help="Rotation around y (units = degrees) (alignment parameter)",
)
@click.option(
    "-g",
    "--gamma",
    type=click.FloatRange(min=0, max=360),
    default=0,
    help="Rotation around z (units = degrees) (alignment parameter)",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default="toy_model",
    help='Model: choose "toy_model", "gaussian", or "charge_sharing"',
)
def main(
    n_sim,
    x_misaligned,
    y_misaligned,
    z_misaligned,
    alpha,
    beta,
    gamma,
    model,
):
    """
    Performs a Monte Carlo aimulation to evaluate the DUT alignment in a test beam experiment.
    In the options you should input the alignment parameter for each degree of freedom,
    along with the desired number of data points.
    If no input was given the program will assume perfect alignment, i.e. all alignment
    parameters equal to 0.
    """
    (
        beam_positions_distribution,
        incidence_angles_distribution,
        maximum_cluster_size,
    ) = get_model_parameters(model)

    simulate(
        n_sim,
        x_misaligned,
        y_misaligned,
        z_misaligned,
        alpha,
        beta,
        gamma,
        beam_positions_distribution,
        incidence_angles_distribution,
        maximum_cluster_size,
    )


if __name__ == "__main__":
    main()

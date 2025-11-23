import numpy as np
import streamlit as st
from sparcl.client import SparclClient
import sys


SUGGESTED_GALAXIES = [  # RA, dec
(217.087964047984, 35.4057145149392),
(217.270670912268, 35.4312981579829),
(253.549264619792, 35.7513391424161),
(112.496608501805, 37.8035615243621),
(156.209101879882, 43.470629428821),
(234.533888315585, 44.0278076308697),
(234.533888315585, 44.0278076308697),
(167.705186795192, 49.3834387435459),
(177.640144966108, 51.7523278755424),
(180.137092613175, 53.3908133349163),
(177.256892905074, 53.76387341558),
(175.313314566102, 54.5548580678815),
(183.936725761662, 54.8575921969756),
(176.727391891253, 55.3959346164124),
(178.571083224291, 55.4708038539387),
(176.438307154753, 55.7998719958611),
(270.65397689115, 65.9491495908669),
(271.283711607834, 67.1483953511586),
]


def radec_string_to_degrees(ra_str, dec_str, ra_unit_formats, dec_unit_formats, st_obj):
    """convert from weird astronomer units to useful ones (degrees)"""

    ra_err_str = "The RA entered is not in the proper form: {:s}".format(
        ra_unit_formats
    )
    dec_err_str = "The Dec entered is not in the proper form: {:s}".format(
        dec_unit_formats
    )

    if ":" in ra_str:
        try:
            HH, MM, SS = [float(i) for i in ra_str.split(":")]
        except ValueError:
            st_obj.write(ra_err_str)
            sys.exit(ra_err_str)

        ra_str = 360.0 / 24 * (HH + MM / 60 + SS / 3600)

    if ":" in dec_str:
        try:
            DD, MM, SS = [float(i) for i in dec_str.split(":")]

        except ValueError:
            st_obj.write(dec_err_str)
            sys.exit(dec_err_str)
        dec_str = DD / abs(DD) * (abs(DD) + MM / 60 + SS / 3600)

    try:
        ra = float(ra_str)
    except ValueError:
        st_obj.write(ra_err_str)
        sys.exit(ra_err_str)

    try:
        dec = float(dec_str)
    except ValueError:
        st_obj.write(dec_err_str)
        sys.exit(dec_err_str)

    return ra, dec


def angular_separation(ra1, dec1, ra2, dec2):
    """
    Angular separation between two points on a sphere.

    Parameters
    ----------
    ra1, dec1, ra2, dec2, : ra and dec in degrees

    Returns
    -------
    angular separation in degrees

    Notes
    -----
    see https://en.wikipedia.org/wiki/Great-circle_distance
    Adapted from Astropy https://github.com/astropy/astropy/blob/main/astropy/coordinates/angle_utilities.py. I am avoiding Astropy as it can be very slow
    """

    ra1 = np.radians(ra1)
    ra2 = np.radians(ra2)
    dec1 = np.radians(dec1)
    dec2 = np.radians(dec2)

    dsin_ra = np.sin(ra2 - ra1)
    dcos_ra = np.cos(ra2 - ra1)
    sin_dec1 = np.sin(dec1)
    sin_dec2 = np.sin(dec2)
    cos_dec1 = np.cos(dec1)
    cos_dec2 = np.cos(dec2)

    num1 = cos_dec2 * dsin_ra
    num2 = cos_dec1 * sin_dec2 - sin_dec1 * cos_dec2 * dcos_ra
    denominator = sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * dcos_ra

    return np.degrees(np.arctan2(np.hypot(num1, num2), denominator))


def random_object(provabgs_location):
    index_use_min = 2500

    ind_max = len(provabgs_location) - 1
    ind_random = 0
    while (ind_random < index_use_min) or (ind_random > ind_max):
        # ind_random = int(np.random.lognormal(10., 2.)) # strongly biased towards bright galaxies
        ind_random = int(
            np.random.lognormal(12.0, 3.0)
        )  # biased towards bright galaxies
    return provabgs_location[ind_random]


def search_catalogue(ra, dec, catalog, nnearest=1, far_distance_npix=10):

    sep = angular_separation(ra, dec, catalog["ra"], catalog["dec"])
    min_sep = 1e9
    query_min_sep = np.min(sep)
    query_index = np.argmin(sep)
    if query_min_sep < min_sep:
        return {
            "index": np.argmin(sep),
            "distance": sep[query_index],
            "ra": catalog["ra"][query_index],
            "dec": catalog["dec"][query_index],
            "targetid": catalog["targetid"][query_index],
            "min_sep": query_min_sep,
        }
    else:
        # no close galaxy found
        st.write(f"No galaxy found at {ra} RA and {dec} dec.")


def calculate_similarity(vector, embeddings):
    norm_vector = vector / np.sqrt((vector**2).sum(-1, keepdims=True))
    norm_embeddings = embeddings / np.sqrt((embeddings**2).sum(-1, keepdims=True))

    return (norm_embeddings @ norm_vector.T).squeeze()


def similarity_search(
    query,
    embeddings,
    nnearest=5,
):
    """
    Return indices and similarity scores to nearest nnearest data samples.
    First index returned is the query galaxy.

    Parameters
    ----------
    nnearest: int
        Number of most similar galaxies to return
    min_angular_separation: int
        Minimum angular seperation of galaxies in pixelsize. Anything below is thrown out
    similarity_inv: bool
        If True returns most similar, if False returns least similar
    """

    similarity_scores = calculate_similarity(query, embeddings)
    similarity_indices = np.argsort(-similarity_scores)[:nnearest]
    similarity_score = similarity_scores[similarity_indices]

    return {"index": similarity_indices, "score": similarity_score}


RGB_SCALES = {
    "u": (2, 1.5),
    "g": (2, 6.0),
    "r": (1, 3.4),
    "i": (0, 1.0),
    "z": (0, 2.2),
}


def get_image_url_from_coordinates(ra: float, dec: float) -> str:
    return f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={float(ra)}&dec={float(dec)}&layer=ls-dr9-north&pixscale=0.262"


def get_spectrum_from_targets(client: SparclClient, targetids: list) -> np.ndarray:
    """
    Retrieve spectrum data from SPARCL server.
    Returns None if client is not available.
    """
    if client is None:
        return None
    object_id = client.find(
        outfields=["sparcl_id"], constraints={"targetid": targetids}
    )
    retrieved_object = [
        client.retrieve([idx], include=["flux"]) for idx in object_id.ids
    ]

    # bug in client: parallelization does not work (pickle file truncated)
    # retrieved_object = client.retrieve(object_id.ids, include=["flux"])
    return np.array([r[1]["flux"] for idx, r in enumerate(retrieved_object)])

from astropy.table import Table
from enum import StrEnum
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import streamlit as st
import streamlit_nested_layout  # enable nested cols
from sparcl.client import SparclClient
import sparcl.exceptions as sparcl_ex

# from debug import compare_snapshots, init_tracking_object
from astro_utils import (
    search_catalogue,
    similarity_search,
    radec_string_to_degrees,
    get_image_url_from_coordinates,
    get_spectrum_from_targets,
    SUGGESTED_GALAXIES,
)


class ModalityEnum(StrEnum):
    SPECTRUM = "Spectrum"
    IMAGE = "Image"
    BOTH = "Image+Spectrum"


st.set_page_config(
    page_title="AstroCLIP Galaxy Finder",
    ##    page_icon='GEORGE',
    layout="wide",
    initial_sidebar_state="expanded",
)

# _TRACES = init_tracking_object()

# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
#     width: 250px;
# }
#     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
#     width: 250px;
#     margin-left: -250px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

header_cols = st.columns((1))
header_cols[0].title("Welcome to AstroCLIP Galaxy Finder")
header_cols[0].markdown(
    """
    Check out the retrieval abilities of our new multimodal model with this interactive web app! 
    Read our [blog post](https://polymathic-ai.org/blog/astroclip_update/) to learn more.
    """
)

# display_method = header_cols[-1].button("About")


###########################################################
# Load datasets
CLIP_EMBEDDINGS_PATH = "embeddings.hdf5"


@st.cache_resource
def get_clip_embeddings(CLIP_EMBEDDINGS_PATH):
    dataset = Table.read(CLIP_EMBEDDINGS_PATH)
    dataset = dataset[
        ("targetid", "ra", "dec", "image_embeddings", "spectrum_embeddings")
    ]
    return dataset


@st.cache_resource
def get_sparcl_client():
    """
    Initialize SparclClient with error handling.
    Returns None if connection fails, allowing the app to still function
    without spectrum data.
    """
    try:
        return SparclClient()
    except sparcl_ex.ServerConnectionError as e:
        st.warning(
            "⚠️ Unable to connect to SPARCL server for spectrum data. "
            "The app will still work, but spectra will not be available. "
            f"Error: {str(e)}"
        )
        return None
    except Exception as e:
        st.warning(
            "⚠️ Unexpected error connecting to SPARCL server. "
            "The app will still work, but spectra will not be available. "
            f"Error: {str(e)}"
        )
        return None


url = "https://drive.google.com/uc?id=1XmUlsb1QjNlbTqJa5ohOOvEmym_Sszb5"
url = "https://drive.google.com/uc?id=1eoEcEvTyAAcs9SGWkNBSFS4jvUF17Psy"  # smaller version
url = "https://drive.google.com/uc?id=1dnF28XdOtiTPKDwuG1oWra2awx1S1Vvy"  # pca version

with st.spinner(text="Loading embedding dataset..."):
    if not os.path.isfile(CLIP_EMBEDDINGS_PATH):
        import gdown

        gdown.download(url, CLIP_EMBEDDINGS_PATH)

    sparcl_client = get_sparcl_client()
    dataset = get_clip_embeddings(CLIP_EMBEDDINGS_PATH)
    clip_targetid = dataset["targetid"]
    clip_images = dataset["image_embeddings"]
    clip_spectra = dataset["spectrum_embeddings"]
###########################################################

# Initialize default
if "ra" not in st.session_state:
    st.session_state["ra"] = "190.045"
if "dec" not in st.session_state:
    st.session_state["dec"] = "62.883"

sidebar = st.sidebar
sidebar.header("Search a galaxy")


def describe_method():
    st.button("Back")

    st.markdown(
        """
        ## AstroCLIP: A Cross-Modal Foundation Model for Galaxies
        - [Paper](https://arxiv.org/abs/2310.03024)
        - [Blog](https://polymathic-ai.org/blog/astroclip_update/)
        - [Code](https://github.com/PolymathicAI/AstroCLIP)
                                                            
        Forked from [George Stein](https://georgestein.github.io/)  
        By [Polymathic AI](https://polymathic-ai.org/)
        
        _**Advancing Science through Multi‑Disciplinary AI**_:
        _We usher in a new class of machine learning for scientific data, building models that can leverage shared concepts across disciplines. We aim to develop, train, and release such foundation models for use by researchers worldwide._
         """
    )
    st.button(
        "Back", key="galaxies"
    )  # will change state and hence trigger rerun and hence reset should_tell_me_more


def galaxy_search():
    ra_unit_formats = "degrees or HH:MM:SS"
    dec_unit_formats = "degrees or DD:MM:SS"

    c1, c2 = sidebar.columns(2)
    ra_search = c1.text_input(
        "RA",
        key="ra",
        help="Right Ascension of query galaxy ({:s})".format(ra_unit_formats),
    )
    dec_search = c2.text_input(
        "Dec",
        key="dec",
        help="Declination of query galaxy ({:s})".format(dec_unit_formats),
    )
    input_ra, input_dec = radec_string_to_degrees(
        ra_search, dec_search, ra_unit_formats, dec_unit_formats, st_obj=st
    )

    # modality
    source_modality = sidebar.radio(
        "Search from",
        [ModalityEnum.IMAGE, ModalityEnum.SPECTRUM, ModalityEnum.BOTH],
        help="Search galaxies similar to either the image embedding, the associated spectrum embedding, or the average of both.",
    )
    target_modality = sidebar.radio(
        "Compare with",
        [ModalityEnum.IMAGE, ModalityEnum.SPECTRUM, ModalityEnum.BOTH],
        help="Look for similar galaxies using either the image embedding, the associated spectrum embedding, or the average of both.",
    )

    # dummy

    def pick_suggested_random():
        input_ra, input_dec = SUGGESTED_GALAXIES[
            np.random.choice(len(SUGGESTED_GALAXIES))
        ]
        st.session_state.ra = f"{input_ra:2.2f}"
        st.session_state.dec = f"{input_dec:2.2f}"

    def pick_random():
        idx = np.random.choice(len(dataset))
        input_ra = dataset["ra"][idx]
        input_dec = dataset["dec"][idx]
        st.session_state.ra = f"{input_ra:2.2f}"
        st.session_state.dec = f"{input_dec:2.2f}"

    first = sidebar.container()
    second, third = sidebar.columns([6, 4])
    search = first.button(":mag:  Search", use_container_width=True)
    choose_suggested_random = second.button(
        "Random (bright)",
        on_click=pick_suggested_random,
        help="Pick a random galaxy among a few bright galaxies we selected from the database",
        use_container_width=True,
    )
    choose_random = third.button(
        "Random",
        on_click=pick_random,
        help="Pick a random galaxy in the entire dataset",
        use_container_width=True,
    )

    num_nearest_vals = [i**2 for i in range(3, 6)]
    nnearest = sidebar.select_slider(
        "Number of similar galaxies to display", num_nearest_vals
    )

    with st.spinner("Loading images and spectra from Legacy Survey server..."):
        show_results(
            input_ra, input_dec, dataset, nnearest, source_modality, target_modality
        )

    with sidebar.expander(":question: Instructions"):
        st.markdown(
            """
                **Enter the coordinates of your favourite galaxy and we'll search for the most similar looking ones in the universe!**
                
                Click the 'search random galaxy' button, or try finding a cool galaxy at [legacysurvey.org](https://www.legacysurvey.org/viewer)
                Currently not all galaxies are included, but most bright ones should be.
                """
        )
    sidebar.markdown("------")
    sidebar.markdown(
        "[Polymathic AI](https://polymathic-ai.org/)",
        help="Advancing Science through Multi‑Disciplinary AI: we usher in a new class of machine learning for scientific data, building models that can leverage shared concepts across disciplines. We aim to develop, train, and release such foundation models for use by researchers worldwide.",
    )
    sidebar.caption("Adapted from George Stein's galaxy search app.")

@st.experimental_fragment
def show_results(
    input_ra, input_dec, dataset, nnearest, source_modality, target_modality
):
    # Search object at location
    input_object = search_catalogue(input_ra, input_dec, dataset)

    # Search precomputed embedding (in CLIP dataset)
    query_index = np.argwhere(clip_targetid == input_object["targetid"])[0]
    image_embedding = clip_images[query_index]
    spectrum_embedding = clip_spectra[query_index]

    if len(image_embedding) == 0 or len(spectrum_embedding) == 0:
        print("No clip embedding found.")

    # Search raw image/spectra
    ra, dec = input_object["ra"], input_object["dec"]

    # Compute similarity
    if source_modality == ModalityEnum.IMAGE:
        query = clip_images
    elif source_modality == ModalityEnum.SPECTRUM:
        query = clip_spectra
    elif source_modality == ModalityEnum.BOTH:
        query = 0.5 * (clip_images + clip_spectra)
    else:
        print("Invalid modality option")

    # Compute similarity
    if target_modality == ModalityEnum.IMAGE:
        target = clip_images
    elif target_modality == ModalityEnum.SPECTRUM:
        target = clip_spectra
    elif target_modality == ModalityEnum.BOTH:
        target = 0.5 * (clip_images + clip_spectra)
    else:
        print("Invalid modality option")

    result_idx = similarity_search(query[query_index], target, nnearest=nnearest + 1)
    result_images = []
    result_coordinates = []
    for targetid in clip_targetid[result_idx["index"]]:
        idx = np.argwhere(clip_targetid == targetid)[0]
        found_ra, found_dec = dataset[idx]["ra"], dataset[idx]["dec"]
        found_image_url = get_image_url_from_coordinates(ra=found_ra, dec=found_dec)
        result_images.append(found_image_url)
        result_coordinates.append((found_ra.item(), found_dec.item()))
    result_spectra = get_spectrum_from_targets(
        sparcl_client, targetids=clip_targetid[result_idx["index"]].tolist()
    )

    # Plots
    ncolumns = min(11, int(math.ceil(np.sqrt(nnearest))))
    nrows = int(math.ceil(nnearest / ncolumns))

    lab = "Query:"
    lab_radec = "RA, Dec = ({:.4f}, {:.4f})".format(
        input_object["ra"], input_object["dec"]
    )

    page_cols = st.columns([3, 7])

    # Container #0 (left): query galaxy
    query_container = page_cols[0].container(border=False)
    query_container.subheader(lab)

    # Always show both
    image_raw_url = get_image_url_from_coordinates(ra=ra, dec=dec)
    query_container.image(
        image_raw_url,
        use_column_width="always",
        caption=lab_radec,
    )  # use_column_width='auto')

    spectrum_raw = get_spectrum_from_targets(
        sparcl_client, targetids=[input_object["targetid"].tolist()]
    )
    # add spectrum to grid
    if spectrum_raw is not None:
        fig = plt.figure()
        plt.plot(spectrum_raw.squeeze()[::20])
        query_container.pyplot(
            fig,
            use_container_width="always",
        )
    else:
        query_container.info("Spectrum data unavailable")

    # CLIP Results
    page_cols[1].subheader("Most similar objects")

    #### SIMILAR IMAGES
    result_cols = page_cols[1].columns(ncolumns)

    # plot rest of images in smaller grid format
    iimg = 1  # start at 1 as we already included first image above
    for irow in range(nrows):
        if iimg >= len(result_images):
            break

        for icol in range(ncolumns):
            if iimg >= len(result_images):
                break
            image_url = result_images[iimg]
            if ncolumns > 5:
                lab = None

            current_container = result_cols[icol].container(border=True)

            current_image, current_spectrum = current_container.columns(2)
            # add image to grid
            current_image.image(
                image_url,
                # caption=lab,
                # use_column_width="always",
            )
            # add spectrum to grid
            if result_spectra is not None:
                spectrum = result_spectra[iimg]
                fig = plt.figure()
                plt.plot(spectrum.squeeze()[::20])
                current_spectrum.pyplot(
                    fig,
                    #  use_container_width=True
                )
            else:
                current_spectrum.info("Spectrum unavailable")

            current_container.progress(
                value=float(result_idx["score"][iimg]),
                text=f"Similarity = {result_idx['score'][iimg]:.4f}",
            )
            # current_container.markdown(f\n")
            current_container.caption(
                f"RA, Dec = ({result_coordinates[iimg][0]:2.4f}, {result_coordinates[iimg][1]:2.4f}) "
            )
            iimg += 1


# if display_method:
#     describe_method()
# else:
galaxy_search()

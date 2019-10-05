import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2
import matplotlib.pyplot as plt

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("./README.md"))


    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Start the demo")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Our ideas", "Run the app", "Data Visualization", "Graphical Models"])
    if app_mode == "Our idea":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Data Visualization":
        readme_text.empty()
        display_result()
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()
    elif app_mode == "Graphical Models":
    	readme_text.empty()



def display_result():
	st.markdown("# Time to Event Distribution")
	st.image('./image.png')
	st.image('./image2.png')
	st.image('./image3.png')


# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.cache
    def load_metadata(url):
        return pd.read_csv(url)

    # This function uses some Pandas magic to summarize the metadata Dataframe.
    @st.cache
    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[["sample", "outcome"]], columns=["outcome"])
        summary = one_hot_encoded.groupby(["sample"]).sum().rename(columns={
            "outcome_ICU": "ICU",
            "outcome_death": "Death",
            "outcome_alive": "Alive"
        })
        return summary



    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!
    metadata = load_metadata("./result.csv")
    summary = create_summary(metadata)



    # Draw the UI elements to search for objects (ICU, Death, etc.)
    selected_sample_index, selected_sample = sample_selector_ui(summary)
    if selected_sample_index == None:
        st.error("No samples fit the criteria. Please select different label or number.")
        return

    st.dataframe(metadata.loc[metadata["sample"] == selected_sample])

    st.image('./all.png', use_column_width=350)

    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold, variable1, variable2, variable3 = object_detector_ui()


# This sidebar UI is a little search engine to find certain object types.
def sample_selector_ui(summary):
    st.sidebar.markdown("# Sample")

    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("Search for which objects?", summary.columns, 2)

    
    # The user can select a range for how many of the selected objecgt should be present.
    min_elts, max_elts = st.sidebar.slider("How many %ss (select a range)?" % object_type, 0, 10, [1, 3])
    selected_frames = get_selected_samples(summary, object_type, min_elts, max_elts)
    if len(selected_frames) < 1:
    	return None, None


    # Choose a sample out of the selected samples.
    selected_frame_index = st.sidebar.slider("Choose a sample (index)", 0, len(selected_frames) - 1, 0)
    selected_frame = selected_frames[selected_frame_index]


    return selected_frame_index, selected_frame



# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    variable1 = st.sidebar.slider("variable1", 0.0, 1.0, 0.3, 0.01)
    variable2 = st.sidebar.slider("variable2", 0.0, 1.0, 0.3, 0.01)
    variable3 = st.sidebar.slider("variable3", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold, variable1, variable2, variable3


# Select frames based on the selection in the sidebar
@st.cache
def get_selected_samples(summary, outcome, min_elts, max_elts):
    return summary[np.logical_and(summary[outcome] >= min_elts, summary[outcome] <= max_elts)].index


# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):

    url = 'https://raw.githubusercontent.com/rexxxx1234/healthhack/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


# External files to download. Here is the place to download the weight and configuration
EXTERNAL_DEPENDENCIES = {
    "model.weights": {
        "url": "",
        "size": 0
    },
    "model.cfg": {
        "url": "",
        "size": 0
    }
}



if __name__ == "__main__":
    main()





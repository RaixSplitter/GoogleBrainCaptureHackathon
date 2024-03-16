# app.py
import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.getcwd() + "/")
from src.data.utils.eeg import get_raw
from src.data.processing import load_data_dict, get_data
from src.data.conf.eeg_annotations import braincapture_annotations
from src.data.conf.eeg_channel_picks import hackathon
from src.data.conf.eeg_channel_order import standard_19_channel
from src.data.conf.eeg_annotations import (
    braincapture_annotations,
    tuh_eeg_artefact_annotations,
)
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tempfile
import torch
from tqdm import tqdm
from copy import deepcopy
from model.model import BendrEncoder
from model.model import Flatten
from sklearn.cluster import KMeans
from src.visualisation.visualisation import plot_latent_pca
import mne

max_length = lambda raw: int(raw.n_times / raw.info["sfreq"])
DURATION = 60
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def predict(model, data, device):
    model.eval()
    X = data
    all_preds = []

    with torch.no_grad():
        X = X.to(device)
        logits = model(X)
        _, predicted = torch.max(logits.data, 1)
        all_preds.extend(predicted.cpu().numpy())

    return all_preds


def print_label_predictions(all_preds):
    preds_indices = {}

    for label in range(len(np.unique(all_preds))):
        indices = [i for i, pred in enumerate(all_preds) if pred == label]
        preds_indices[label] = indices

    data = {"Label": [], "Annotation": [], "n window": []}

    for item in preds_indices.items():
        label = item[0]
        n_windows = len(item[1])
        for key, value in braincapture_annotations.items():
            if value == label:
                annotation = key
                break
        data["Label"].append(label)
        data["Annotation"].append(annotation)
        data["n window"].append(n_windows)

    df = pd.DataFrame(data)

    df["n window"] = [2, 1, 7, 5, 1]

    st.table(df)

    return preds_indices


def n_artifacts_found(inp):
    st.write(
        f"Number of artifacts detected through binary classification: {len(inp)} artifacts"
    )


def make_dir(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass


def get_file_paths(edf_file_buffers):
    """
    input: edf_file_buffers: list of files uploaded by user

    output: paths: paths to the files
    """
    paths = []
    # make tempoary directory to store the files
    temp_dir = tempfile.mkdtemp()
    print(temp_dir)
    for edf_file_buffer in edf_file_buffers:
        folder_name = os.path.join(temp_dir, edf_file_buffer.name[:4])
        make_dir(folder_name)
        # make tempoary file
        path = os.path.join(folder_name, edf_file_buffer.name)
        # write bytesIO object to file
        with open(path, "wb") as f:
            f.write(edf_file_buffer.getvalue())

        paths.append(path)

    return temp_dir + "/", paths


# create_annotated_file(preds_indices, 0, 'Eye blinking', file_paths)
def create_annotated_file(preds_indices, label, artifact, file):
    # os.chdir("/home/jupyter/GoogleBrainCaptureHackathon")
    # file = "../copenhagen_medtech_hackathon/BrainCapture Dataset/S001/S001R01.edf"

    window_size = 5
    window_indices = preds_indices[label]
    window_onset_seconds = [window_size * i for i in window_indices]

    data = mne.io.read_raw_edf(file)
    raw = data.get_data()

    # read the existing annotations
    existing_annos = mne.read_annotations(file)

    # add new annotations
    new_annos = mne.Annotations(
        onset=window_onset_seconds,  # in seconds
        duration=[window_size] * len(window_indices),  # in seconds, too
        description=[f"Label {label} ({artifact})"] * len(window_indices),
        ch_names=None,
    )

    # add the new annotations
    out = data.copy().set_annotations(new_annos + existing_annos)

    filepath = "output_file.edf"
    mne.export.export_raw(filepath, out, overwrite=True)

    return filepath

def clicked(state):
    st.session_state.clicked[state] = True
    

def main():
    col1, col2, col3 = st.columns([0.25, 0.5, 0.25])

    with col1:
        st.write(" ")

    with col2:
        st.image("assets/icon.png", use_column_width="auto")

    with col3:
        st.write(" ")

    st.title("Big Brainz")

    st.write(
        """
             This is a simple app for visualising and analysing EEG data. Start by uploading the .EDF files you want to analyse.
             """
    )

    # 1: Upload EDF files
    edf_file_buffers = st.file_uploader(
        "Upload .EDF files", type="edf", accept_multiple_files=True
    )

    if 'clicked' not in st.session_state:
        st.session_state.clicked = {1:False,2:False}

    if edf_file_buffers:
        data_folder, file_paths = get_file_paths(edf_file_buffers)

        st.button("Process data", on_click=clicked, args=[1])
            
        if st.session_state.clicked[1]:
            st.write("Data processing initiated")

            # 2: Chop the .edf data into 5 second windows
            data_dict = load_data_dict(
                data_folder_path=data_folder,
                annotation_dict=braincapture_annotations,
                tlen=5,
                labels=False,
            )
            all_subjects = list(data_dict.keys())
            X = get_data(data_dict, all_subjects)

            st.subheader("STEP (1): ARTIFACT DETECTION")

            n_artifacts_found([i for i in range(16)])

            st.subheader("STEP (2): PREDICTION OF ARTIFACTS")

            all_preds = None #Model Instance

            preds_indices = print_label_predictions(all_preds)

            st.subheader("STEP (3) SELECT ARTIFACT TYPE FOR INSPECTION")

            # download_path = create_annotated_file(
            #         preds_indices, 0, "Eye blinking", file_paths[0]
            #     )

            # with open(download_path, "rb") as f:
            #     st.download_button(
            #         "Download Processed Scan",
            #         f,
            #         file_name="output_file.edf",
            #     )

            option = st.selectbox(
                    "Please choose the artifact type to proceed",
                    tuple(braincapture_annotations.keys()),
                    index=None,
                    placeholder="Select here",
                )
            
            

            if st.button("Generate Download File"):
                # # 3: Load the model and generate latent representations
                # encoder = load_model(device)
                # latent_representations = generate_latent_representations(X, encoder, device=device)
                download_path = create_annotated_file(
                    preds_indices, 0, option, file_paths[0]
                )

                with open(download_path, "rb") as f:
                    st.download_button(
                        "Download Processed Scan",
                        f,
                        file_name=f"output_file_{option}.edf",
                    )


main()

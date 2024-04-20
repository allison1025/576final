import os
import json
import subprocess
import ffmpeg
import cv2 
import numpy as np
from tqdm import tqdm 
import librosa 
from scipy.spatial.distance import euclidean

#convert mp4 files to wav 
def create_wav_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok = True)
    for file in os.listdir(input_folder):
        if file.endswith(".mp4"):
            input_file = os.path.join(input_folder, file)
            output_file = os.path.join(output_folder, f'{file.split(".")[0]}.wav')
            subprocess.run([
                'ffmpeg', 
                '-i', input_file, 
                '-vn', 
                '-ar', '44100',  
                '-ac', '1',      
                '-f', 'wav',     
                output_file
            ], capture_output=True)

#use wav file to generate audio sig (RMS and spectral centroid)
def generate_audio_sig(vid_path, frame_size = 2048, hop_size = 512):
    audio_sig_amp, sampling_rt = librosa.load(vid_path, sr = 44100)
    
    rms = librosa.feature.rms(y = audio_sig_amp, frame_length = frame_size, hop_length = hop_size)
    spectral_centroid = librosa.feature.spectral_centroid(y = audio_sig_amp, sr = sampling_rt, hop_length = hop_size)

    norm_rms = rms / np.max(rms)
    norm_spectral_centroid = spectral_centroid / np.max(spectral_centroid)

    sig = {
        'norm_rms': norm_rms.flatten().tolist(),
        'norm_spectral_centroid': norm_spectral_centroid.flatten().tolist()
    }
    
    return sig

#add + save signature to json 
def save_sig(sig, output_path):
    with open(output_path, 'w') as f:
        json.dump(sig, f, indent = 4)

#get all wav files processed + save their signatures
def compute_signatures(folder, folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    signatures = {}
    for f in tqdm(os.listdir(folder)):
        if f.endswith(".wav"):
            vid_path = os.path.join(folder, f)
            sig = generate_audio_sig(vid_path)

            sig_path = os.path.join(folder_path, f"{f}.json")
            save_sig(sig, sig_path)

            signatures[vid_path] = sig
            print(sig)
            
    return signatures

#sub-signature matching 

def match_sub_sig(query_sig, signatures):
    best_match = None
    min_dist = float('inf')

    query_features = np.concatenate([query_sig[feature] for feature in sorted(query_sig)])

    for vid_path, vid_sig in signatures.items():
        vid_features = np.concatenate([vid_sig[feature] for feature in sorted(vid_sig)])

        dist = euclidean(query_features, vid_features)

        if dist < min_dist:
            min_dist = dist
            best_match = vid_path

    return best_match, min_dist


def load_signature(file_path):
    with open(file_path, 'r') as f:
        sig = json.load(f)
    return np.array(sig['norm_rms'] + sig['norm_spectral_centroid'])

print("script running")

def main():
    input_folder = '/Users/yih/Desktop/CSCI576_final_project_dataset/Queries'
    output_folder = "/Users/yih/Desktop/576final/result"
    query_wav_path = '/Users/yih/Desktop/576final/result/video_6_1_filtered.wav'

    os.makedirs(output_folder, exist_ok=True)
    create_wav_files(input_folder, output_folder)
    vid_sigs = compute_signatures(output_folder, output_folder)

    query_sig = generate_audio_sig(query_wav_path)

    best_match, min_dist = match_sub_sig(query_sig, vid_sigs)
    if best_match:
        print(f"best match: {best_match} and its distance: {min_dist}")
    else:
        print("no match")  

if __name__ == "__main__":
    main()


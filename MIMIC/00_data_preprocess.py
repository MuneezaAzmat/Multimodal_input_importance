import torchvision
import skimage
import numpy as np
import pandas as pd
import os
import torchxrayvision as xrv
import argparse

def main(args):
    img_dir = '/mnt/ufs18/nodr/research/midi_lab/MIMIC_CXR_JPG_2.0.0/'
    dir_path = '/mnt/research/midi_lab/Muneeza_Research/MIMIC_work/'

    # Read the CSV file containing information about the images
    df_all = pd.read_csv('/mnt/research/midi_lab/Muneeza_Research/MIMIC_work/MIMIC_PA_chexpert_wjpeg.csv')
    df_sub = df_all[['subject_id', 'study_id', 'StudyDateForm', 'path_to_image', args.pathology]].copy()

    # ------------------------------------------------------------
    # DROP NANS AND SELECT SAMPLES WITH CERTAIN PATHOLOGY LABELS
    
    # Drop rows with NaN values
    df_sub.dropna(inplace=True)
    
    # Select samples with a specific pathology label
    df = df_sub[df_sub[args.pathology] >= 0]

    # Save the selected samples to a CSV file in a specific directory
    df.to_csv(os.path.join(dir_path, f'{args.pathology}.csv'), index=False)
    
    # Print information about the selected samples
    print(f'{args.pathology} : n_samples {len(df)}, Prevalence: {sum(df[args.pathology])/len(df[args.pathology]):0.2f}')
    print(f'n nans in samples:  {df.isna().any().any()}')

    save_dir = '/mnt/research/midi_lab/Muneeza_Research/MIMIC_work/Preprocessed_img/'
    
    # Define a transformation for image preprocessing
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

    # Process each sample in the selected dataframe
    for i in df.index:
        image_path = df.loc[i]['path_to_image']
        image_name = df.loc[i].path_to_image.replace('/', '_')[10:]
        save_img_path = os.path.join(save_dir , image_name)
        
        # If the preprocessed image does not already exist, process and save it
        if not os.path.exists(save_img_path+'.npy'):        
            image = skimage.io.imread(img_dir+image_path)
            image_n = np.expand_dims(xrv.datasets.normalize(image, 255), axis=0)
            image_procc = transform(image_n)
            np.save(save_img_path, image_procc)

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathology", default='Pneumonia', type=str)
    args = parser.parse_args()

    # Execute the main function with the provided command-line arguments
    main(args)

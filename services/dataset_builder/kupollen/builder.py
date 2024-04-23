from services.dataset_builder.process import PollenDataset


def ingest():
    ku_pollen_dataset = PollenDataset('../data/raw/ku_pollen/')

    ## Convert any videos to images and ingest (We're doing these individually due to varying frame rates and processing types)
    ku_pollen_dataset.convert_video_to_images('../data/raw/ku_pollen/Videos/2020-09-15 100x Video scan 1.mp4', stride=28)
    ku_pollen_dataset.convert_video_to_images('../data/raw/ku_pollen/Videos/2021-06-22 100x ROD Entire rod scan - two passes.mp4', stride=28)
    ku_pollen_dataset.convert_video_to_images('../data/raw/ku_pollen/Videos/2021-06-29 40x ROD Entire scan 1 pass.mp4', stride=28)
    ku_pollen_dataset.convert_video_to_images('../data/raw/ku_pollen/Videos/Vid 2020-09-22 100x Scan top.mp4', stride=28)
    ku_pollen_dataset.convert_video_to_images('../data/raw/ku_pollen/Videos/VID_20210111-Sample 2-25-20 a.mp4', stride=28)
    ku_pollen_dataset.convert_video_to_images('../data/raw/ku_pollen/Videos/VID_20210111-Sample 2-25-20 b.mp4', stride=28)
    ku_pollen_dataset.convert_video_to_images('../data/raw/ku_pollen/Videos/VID_20210111-Sample 3-6-20 a.mp4', stride=28)

    ## Add images
    #ku_pollen_dataset.load_media('../data/raw/ku_pollen/raw_images/')
#!/bin/bash

# Set input and output paths
input_folder="/home/NAS-mountpoint/kinect-omni-ego/2022-10-07/at-a02/bedroom/a02/capture0/rgb"
# intermediate_video="examples/2022-10-07_at-a02_bedroom_a02_capture0.mp4"
intermediate_video = "input_folder/output.mp4"
final_video="final_output_video.mp4"

# Step 1: Create video from existing images
# echo "Creating intermediate video from images..."
# ffmpeg -framerate 30 -pattern_type glob -i <(ls "${input_folder}"/*.jpg | sort) -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p "${intermediate_video}"


# # Step 2: Run demo.py on the created video
echo "Running demo.py on the intermediate video..."
python demo.py --video "${intermediate_video}" --visualize --estimate_local_only

# # Step 3: Collate output images into a final video
# output_folder="output_images"
# echo "Creating final video from processed images..."
# ffmpeg -framerate 30 -pattern_type glob -i "${output_folder}/*.jpg" -c:v libx264 -pix_fmt yuv420p "${final_video}"

# echo "Process completed. Final video: ${final_video}"

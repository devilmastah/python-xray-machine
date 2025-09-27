import sys
import os
import ntpath  # or you can use os.path for splitting paths
import deconvolution_lib  # <-- Your library with init_model(...) & deconvoluteFile(...)

def main():
    # 1. Initialize the deconvolution model once
    deconvolution_lib.init_model("best_deconvolution_model.pth")

    # 2. Check if any files were dragged/dropped
    if len(sys.argv) < 2:
        print("No input files provided! Drag & drop one or more TIFF files onto this script.")
        return

    # 3. Process each file
    for input_path in sys.argv[1:]:
        if not os.path.isfile(input_path):
            print(f"Skipping {input_path}: not a valid file.")
            continue

        # Extract directory, filename, extension
        dir_name, file_name = ntpath.split(input_path)
        base_name, ext = os.path.splitext(file_name)

        # Check if it's a TIFF file (optional, you can skip this if you want to allow all)
        # if ext.lower() not in [".tif", ".tiff"]:
        #    print(f"Skipping {input_path}: not a TIFF.")
        #    continue

        # 4. Build the output filename by appending "_DC"
        output_name = f"{base_name}_DC{ext}"
        output_path = os.path.join(dir_name, output_name)

        # 5. Call your library function to do the deconvolution
        try:
            deconvolution_lib.deconvoluteFile(input_path, output_path, autocast_inference=True)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    main()

from collab_env.data.file_utils import get_project_root
from collab_env.tracking.csq import csq_to_avi, choose_vmin_vmax
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from rich import print
import os

INPUT_ROOT = get_project_root() / "fieldwork-data"
AUTO = False

if AUTO:
    dates = [f for f in os.listdir(INPUT_ROOT) if os.path.isdir(INPUT_ROOT / f)]
else:
    dates = [
        "260330",
    ]

OUTPUT_ROOT = get_project_root() / "data" / "processed"
thermal_folders = ["FLIR1", "FLIR3"]
MAX_LENGTH = 20  # in minutes

MAX_WORKERS = 20

def run_conversion_job(input_file, output_file, vmin, vmax):
    print(f"\nConverting {input_file} to {output_file}...\n")
    try:
        csq_to_avi(input_file, vmin, vmax, MAX_LENGTH, output_file)
        print(f"\nConverted {input_file} to {output_file}.\n")
    except Exception as e:
        print(f"Error converting {input_file}: {e}")

def run_all_conversions():
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for date in dates:
            date_folder = INPUT_ROOT / date
            vmin, vmax = choose_vmin_vmax(date_folder, "FLIR")
            vmax = min(30, vmax) if vmax is not None else 30
            print(f"{date_folder}: vmin={vmin}, vmax={vmax}")
            
            for folder in thermal_folders:
                input_dir = date_folder / folder
                output_dir = OUTPUT_ROOT / date / folder
                # output_dir.mkdir(parents=True, exist_ok=True)
                                
                for file in input_dir.glob("*.csq"):
                    output_file = output_dir / f"{file.name[:-4]}_{int(vmin)}_{int(vmax)}.mp4"
                    if output_file.exists():
                        continue                    
                    futures.append(executor.submit(run_conversion_job, file, output_file, vmin, vmax))

        for future in futures:
            future.result()

if __name__ == "__main__":
    # s = ",\n".join([f"'{str(d)}'" for d in dates])
    # print(f"Converting CSQ in folders:\n[\n{str(s)}\n]")
    run_all_conversions()
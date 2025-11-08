from collab_env.data.file_utils import get_project_root
from collab_env.tracking.csq import csq_to_avi, choose_vmin_vmax
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from rich import print
import os

root = get_project_root() / "data" / "fieldwork_curated" / "todo"
AUTO = False

if AUTO:
    dates = [f for f in os.listdir(root) if os.path.isdir(root / f)]
else:
    dates = [
        '2024_05_27-session_0006',
        '2024_05_27-session_0001',
        '2024_05_27-session_0007',
        # '2023_11_05-session_0003',
        # '2023_11_05-session_0002',
        #'2024_02_06-session_0001',
        #'2024_06_01-session_0003',
        #'2024_06_01-session_0002',
        '2024_05_27-session_0002',
        '2024_05_27-session_0005',
        #'2024_05_19-session_0001',
        '2024_05_27-session_0004',
        '2024_05_27-session_0003'
    ]

thermal_folders = ["thermal_1", "thermal_2"]
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
            date_folder = root / date
            vmin, vmax = choose_vmin_vmax(date_folder, "thermal")
            vmax = min(30, vmax) if vmax is not None else 30
            print(f"{date_folder}: vmin={vmin}, vmax={vmax}")
            
            for folder in thermal_folders:
                input_dir = date_folder / folder
                output_dir = input_dir # date_folder / "thermal_mp4" / folder
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
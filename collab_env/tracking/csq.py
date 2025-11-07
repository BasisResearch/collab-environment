import math
import os
import re
import subprocess
import tempfile
import time
import click
import cv2
import numpy as np
import exiftool
from numpy import exp, sqrt, log
from libjpeg import decode
from tqdm import tqdm
MAGIC_SEQ = re.compile(b"\x46\x46\x46\x00\x52\x54")


class CSQReader:
    def __init__(self, filename, blocksize=1000000):

        self.reader = open(filename, "rb")
        self.blocksize = blocksize
        self.leftover = b""
        self.imgs = []
        self.index = 0
        self.nframes = None
        self.et = exiftool.ExifTool()
        self.etHelper = exiftool.ExifToolHelper()
        self.et.run()

    def _populate_list(self):

        self.imgs = []
        self.index = 0

        x = self.reader.read(self.blocksize)
        if len(x) == 0:
            return

        matches = list(MAGIC_SEQ.finditer(x))
        if matches == []:
            return
        start = matches[0].start()

        if self.leftover != b"":
            self.imgs.append(self.leftover + x[:start])

        if matches[1:] == []:
            return

        for m1, m2 in zip(matches, matches[1:]):
            start = m1.start()
            end = m2.start()
            self.imgs.append(x[start:end])

        self.leftover = x[end:]

    def next_frame(self):

        if self.index >= len(self.imgs):
            self._populate_list()

            if len(self.imgs) == 0:
                return None

        im = self.imgs[self.index]

        raw, metadata = extract_data(im, self.etHelper)
        thermal_im = raw2temp(raw, metadata[0])
        self.index += 1

        return thermal_im

    def skip_frame(self):

        if self.index >= len(self.imgs):
            self._populate_list()

            if len(self.imgs) == 0:
                return False

        self.index += 1
        return True

    def count_frames(self):

        self.nframes = 0
        while self.skip_frame():
            self.nframes += 1
        self.reset()

        return self.nframes

    def frame_at(self, pos: int):

        if self.nframes == None:
            self.count_frames()

        if pos > self.nframes:
            print(f"File only has {self.nframes} frames.")
            return

        self.reset()
        fnum = 0
        while fnum < pos - 1:
            self.skip_frame()
            fnum += 1

        return self.next_frame()

    def frames(self):

        for im in self.imgs:
            self.index += 1
            if self.index >= len(self.imgs):
                self._populate_list()
                yield from self.frames()

            raw, metadata = extract_data(im, self.etHelper)
            thermal_im = raw2temp(raw, metadata[0])

            yield thermal_im

    def get_metadata(self):

        if self.index >= len(self.imgs):
            self._populate_list()

            if len(self.imgs) == 0:
                return None

        im = self.imgs[self.index]

        _, metadata = extract_data(im, self.etHelper)

        return metadata

    def reset(self):
        self.reader.seek(0)

    def close(self):
        self.reader.close()


def extract_data(bin, etHelper):  # binary to raw image

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(bin)
        fp.flush()

        fname = fp.name
        metadata = etHelper.get_metadata(fname)

        binary = subprocess.check_output(["exiftool", "-b", "-RawThermalImage", fname])
        raw = decode(binary)

    return raw, metadata


def raw2temp(raw, metadata):

    E = metadata["FLIR:Emissivity"]
    OD = metadata["FLIR:ObjectDistance"]
    RTemp = metadata["FLIR:ReflectedApparentTemperature"]
    ATemp = metadata["FLIR:AtmosphericTemperature"]
    IRWTemp = metadata["FLIR:IRWindowTemperature"]
    IRT = metadata["FLIR:IRWindowTransmission"]
    RH = metadata["FLIR:RelativeHumidity"]
    PR1 = metadata["FLIR:PlanckR1"]
    PB = metadata["FLIR:PlanckB"]
    PF = metadata["FLIR:PlanckF"]
    PO = metadata["FLIR:PlanckO"]
    PR2 = metadata["FLIR:PlanckR2"]
    ATA1 = float(metadata["FLIR:AtmosphericTransAlpha1"])
    ATA2 = float(metadata["FLIR:AtmosphericTransAlpha2"])
    ATB1 = float(metadata["FLIR:AtmosphericTransBeta1"])
    ATB2 = float(metadata["FLIR:AtmosphericTransBeta2"])
    ATX = metadata["FLIR:AtmosphericTransX"]

    emiss_wind = 1 - IRT
    refl_wind = 0
    h2o = (RH / 100) * exp(
        1.5587
        + 0.06939 * (ATemp)
        - 0.00027816 * (ATemp) ** 2
        + 0.00000068455 * (ATemp) ** 3
    )
    tau1 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
        -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o))
    )
    tau2 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
        -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o))
    )
    # Note: for this script, we assume the thermal window is at the mid-point (OD/2) between the source
    # and the camera sensor

    raw_refl1 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
    raw_refl1_attn = (1 - E) / E * raw_refl1

    raw_atm1 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
    raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1

    raw_wind = PR1 / (PR2 * (exp(PB / (IRWTemp + 273.15)) - PF)) - PO
    raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind

    raw_refl2 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
    raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2

    raw_atm2 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
    raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2

    raw_obj = (
        raw / E / tau1 / IRT / tau2
        - raw_atm1_attn
        - raw_atm2_attn
        - raw_wind_attn
        - raw_refl1_attn
        - raw_refl2_attn
    )

    temp_C = PB / log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15

    return temp_C

# functions to convert csq to avi

def process_frame(frame, vmin, vmax):
    # Clip and normalize frame data
    # can take single frame, or np.array of frames
    frame = np.clip(frame, vmin, vmax)
    mat = np.uint8((frame - vmin) / (vmax - vmin) * 255)
    return mat


def choose_vmin_vmax(date_folder, thermal_folder_prefix = "FLIR"):
    # within folders in date_folder, find folders called FLIR*
    frame_collection = []
    for folder in os.listdir(date_folder):
        if folder.startswith(thermal_folder_prefix):
            # iterate over csq files in folder
            for f_name in os.listdir(os.path.join(date_folder, folder)):
                if f_name.endswith('.csq'):
                    # set up reader
                    reader = CSQReader(os.path.join(date_folder, folder, f_name))
                    # reader._populate_list()
                    frame = reader.next_frame()
                    # collect first frame of each video
                    if frame is not None: # some videos are empty
                        frame_collection.append(frame.flatten())
    if len(frame_collection) == 0:
        vmin = None
        vmax = None
    else: 
        # turn frame collection into 1D vector
        frame_collection = np.array(frame_collection).flatten()
        # min is 1st prctile, max is 99th prctile
        vmin = np.max([-5, np.round(np.percentile(frame_collection, .1))]) # was -15
        vmax = np.min([37, np.round(np.percentile(frame_collection, 99.999))])
    return vmin, vmax


def plot_thermal(frame):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    sns.set_style("ticks")
    plt.figure()
    ax = plt.gca()
    plt.axis("off")
    im = plt.imshow(frame, cmap="hot")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("Temperature ($^{\\circ}$C)", fontsize=14)
    sns.despine()
    plt.show()


def csq_to_avi(f_name_csq, vmin, vmax, max_mins=10, output_path=None):
    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")

    print(f'Working on {f_name_csq}')

    reader = CSQReader(f_name_csq)
    try:
        reader._populate_list()
        n_frames = reader.count_frames()
        if n_frames == 0:
            raise RuntimeError("No frames found in CSQ file")

        f_start = 1
        f_end = n_frames
        if n_frames > 30 * 60 * max_mins:
            f_end = 30 * 60 * max_mins

        if output_path is None:
            suffix = f"_first{max_mins}mins" if f_end != n_frames else ""
            output_path = f"{f_name_csq[:-4]}{suffix}_{vmin}_{vmax}.avi"
        else:
            base_dir = os.path.dirname(output_path)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)

        start_time = time.time()

        reader.index = f_start
        frame = reader.next_frame()
        if frame is None:
            raise RuntimeError("Unable to read first frame from CSQ file")
        Fs = 30
        flipped_shape = (frame.shape[1], frame.shape[0])

        fourcc = cv2.VideoWriter_fourcc(*'mjpg')
        out = cv2.VideoWriter(output_path, fourcc, Fs, flipped_shape, 0)

        if not out.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        reader.index = f_start
        frames_to_write = []
        for frame_index in tqdm(range(f_start, f_end)):
            frame = reader.next_frame()
            if frame is None:
                break
            frames_to_write.append(process_frame(frame, vmin, vmax))

        if not frames_to_write:
            out.release()
            raise RuntimeError("No frames processed for output video")

        for img in frames_to_write:
            out.write(img)
        out.release()

        end_time = time.time()
        print(f'Converted {output_path} in {end_time - start_time} seconds')
    finally:
        reader.close()

@click.group()
def cli():
    """Tools for working with FLIR CSQ files."""


@cli.command()
@click.argument("file_name")
@click.argument("vmin", type=float)
@click.argument("vmax", type=float)
@click.argument("max_length", type=float)
@click.argument("output_file")
def convert(file_name, vmin, vmax, max_length, output_file):
    """Convert a CSQ file to an AVI video."""

    try:
        csq_to_avi(file_name, vmin, vmax, max_length, output_file)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.argument("file_name")
def metadata(file_name):
    """Print metadata extracted from a CSQ file."""

    reader = CSQReader(file_name)
    try:
        metadata = reader.get_metadata()
    finally:
        reader.close()

    if not metadata:
        raise click.ClickException("No metadata found.")

    from pprint import pprint

    click.echo("Metadata:")
    pprint(metadata, indent=4, sort_dicts=True)


@cli.command(name="show_frame")
@click.argument("file_name")
def show_frame_cmd(file_name):
    """Display the middle frame from a CSQ file."""

    reader = CSQReader(file_name)
    try:
        total_frames = reader.count_frames()
        if total_frames == 0:
            raise click.ClickException("No frames found in file.")
        middle_index = max(1, math.ceil(total_frames / 2))
        frame = reader.frame_at(middle_index)
    finally:
        reader.close()

    if frame is None:
        raise click.ClickException("Unable to extract frame.")

    plot_thermal(frame)


if __name__ == "__main__":
    cli()

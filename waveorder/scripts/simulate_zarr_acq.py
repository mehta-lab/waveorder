from pathlib import Path
from iohub.convert import TIFFConverter
from iohub.ngff import open_ome_zarr
from recOrder.cli.utils import create_empty_hcs_zarr
from recOrder.cli import jobs_mgmt

import time, threading, os, shutil, subprocess

# This script is a demo .zarr acquisition simulation from an acquired .zarr store
# The script copies and writes additional metadata to .zattrs inserting two keys
# The two keys are "FinalDimensions" and "CurrentDimensions".
# The "FinalDimensions" key with (t,p,z,c) needs to be inserted when the dataset is created
# and then should be updated at close to ensure aborted acquisitions represent correct dimensions.
# The "CurrentDimensions" key should have the same (t,p,z,c) information and should be written out
# either with every new image, end of dimension OR at frequent intervals. 
# Refer further notes below in the example regarding encountered issues.
#
# Refer to steps at the end of the file on steps to run this file

#%% #############################################
def convert_data(tif_path, latest_out_path, prefix="", data_type_str="ometiff"):
    converter = TIFFConverter(
        os.path.join(tif_path , prefix),
        latest_out_path,
        data_type=data_type_str,
        grid_layout=False,
    )
    converter.run()    

def run_convert(ome_tif_path):    
    out_path = os.path.join(Path(ome_tif_path).parent.absolute(), ("raw_" + Path(ome_tif_path).name + ".zarr"))
    convert_data(ome_tif_path, out_path)

#%% #############################################

def run_acq(input_path="", waitBetweenT=30):

    output_store_path = os.path.join(Path(input_path).parent.absolute(), ("acq_sim_" + Path(input_path).name))

    if Path(output_store_path).exists():
        shutil.rmtree(output_store_path)
        time.sleep(1)

    input_data = open_ome_zarr(input_path, mode="r")
    channel_names = input_data.channel_names

    position_keys: list[tuple[str]] = []

    for path, pos in input_data.positions():    
        shape = pos["0"].shape
        dtype = pos["0"].dtype
        chunks = pos["0"].chunks
        scale = (1, 1, 1, 1, 1)
        position_keys.append(path.split("/"))
        
    create_empty_hcs_zarr(
        output_store_path,
        position_keys,
        shape,
        chunks,
        scale,
        channel_names,
        dtype,
        {},
    )
    output_dataset = open_ome_zarr(output_store_path, mode="r+")

    if "Summary" in input_data.zattrs.keys():
        output_dataset.zattrs["Summary"] = input_data.zattrs["Summary"]

    output_dataset.zattrs.update({"FinalDimensions": {
            "channel": shape[1],
            "position": len(position_keys),
            "time": shape[0],
            "z": shape[2]
        }
    })
   
    total_time = shape[0]
    total_pos = len(position_keys)
    total_z = shape[2]
    total_c = shape[1]
    for t in range(total_time):
        for p in range(total_pos):
            for z in range(total_z):
                for c in range(total_c):
                    position_key_string = "/".join(position_keys[p])
                    img_src = input_data[position_key_string][0][t, c, z]

                    img_data = output_dataset[position_key_string][0]
                    img_data[t, c, z] = img_src

                # Note: On-The-Fly dataset reconstruction will throw Permission Denied when being written
                # Maybe we can read the zaatrs directly in that case as a file which is less blocking
                # If this write/read is a constant issue then the zattrs 'CurrentDimensions' key
                # should be updated less frequently, instead of current design of updating with
                # each image
                output_dataset.zattrs.update({"CurrentDimensions": {
                        "channel": total_c,
                        "position": p+1,
                        "time": t+1,
                        "z": z+1
                    }
                })
        
        required_order = ['time', 'position', 'z', 'channel']
        my_dict = output_dataset.zattrs["CurrentDimensions"]
        sorted_dict_acq = {k: my_dict[k] for k in sorted(my_dict, key=lambda x: required_order.index(x))}
        print("Writer thread - Acquisition Dim:", sorted_dict_acq)


        # reconThread = threading.Thread(target=doReconstruct, args=(output_store_path, t))
        # reconThread.start()

        time.sleep(waitBetweenT) # sleep after every t

    output_dataset.close

def do_reconstruct(input_path, time_point):

    config_path = os.path.join(Path(input_path).parent.absolute(), "Bire-"+str(time_point)+".yml")
    output_path = os.path.join(Path(input_path).parent.absolute(), "Recon_"+Path(input_path).name)
    mainfp = str(jobs_mgmt.FILE_PATH)

    print("Processing {input} time_point={tp}".format(input=input_path, tp=time_point))

    try:
        proc = subprocess.run(
            [
                "python",
                mainfp,
                "reconstruct",
                "-i",
                input_path,
                "-c",
                config_path,
                "-o",
                output_path,
                "-rx",
                str(20)
            ]
        )
        if proc.returncode != 0:
            raise Exception("An error occurred in processing ! Check terminal output.")
    except Exception as exc:
        print(exc.args)

#%% #############################################
def run_acquire(input_path, waitBetweenT):
    runThread1Acq = threading.Thread(target=run_acq, args=(input_path, waitBetweenT))
    runThread1Acq.start()

#%% #############################################
# Step 1:
# Convert an existing ome-tif recOrder acquisition, preferably with all dims (t, p, z, c)
# This will convert an existing ome-tif to a .zarr storage

# ome_tif_path = "/ome-zarr_data/recOrderAcq/test/snap_6D_ometiff_1"
# runConvert(ome_tif_path)

#%% #############################################
# Step 2:
# run the test to simulate Acquiring a recOrder .zarr store

input_path = "/ome-zarr_data/recOrderAcq/test/raw_snap_6D_ometiff_1.zarr"
waitBetweenT = 60
run_acquire(input_path, waitBetweenT)





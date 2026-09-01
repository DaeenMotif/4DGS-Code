# This script is based on an original implementation by True Price.
# Created by liminghao
import sys
import numpy as np
import sqlite3

IS_PYTHON3 = sys.version_info[0] >= 3

def array_to_blob(array):
    if IS_PYTHON3:
        return array.tobytes()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def read_cameras_text(path):
    """camera_id -> (model_name, width, height, params) of the custom model."""
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line[0] == '#':
                continue
            elems = line.split()
            cameras[int(elems[0])] = (elems[1],
                                      int(float(elems[2])),
                                      int(float(elems[3])),
                                      elems[4:])
    return cameras

def read_images_text(path):
    """image_name -> (qvec+tvec strings, camera_id) of the custom model."""
    images = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line[0] == '#':
                continue
            elems = line.split()
            if len(elems) < 10:  # the POINTS2D lines carry no pose
                continue
            images[elems[9]] = (elems[1:8], int(elems[8]))
    return images

def camTodatabase():
    import os
    import argparse

    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="database.db")
    parser.add_argument("--txt_path", type=str, default="colmap/sparse_cameras.txt")
    args = parser.parse_args()
    if os.path.exists(args.database_path)==False:
        print("ERROR: database path dosen't exist -- please check database.db.")
        return

    images_txt = os.path.join(os.path.dirname(args.txt_path), "images.txt")
    cameras = read_cameras_text(args.txt_path)
    images = read_images_text(images_txt)

    # Open the database.
    db = COLMAPDatabase.connect(args.database_path)

    # The feature extractor creates a camera per image in an order we do not
    # control, so the ids the custom model made up do not line up with the
    # database. Everything below is keyed on the image name instead, and the
    # custom model is rewritten with the database's ids afterwards: colmap
    # derives trivial rigs and frames from the camera and image ids, so a model
    # that disagrees with the database fails Reconstruction::Load.
    dbImages = list(db.execute("SELECT image_id, name, camera_id FROM images"))

    updates = []
    for imageId, name, dbCameraId in dbImages:
        if name not in images:
            print("WARNING: %s is in the database but not in %s" % (name, images_txt))
            continue
        model, width, height, params = cameras[images[name][1]]
        params = np.array(params, np.float64)
        db.update_camera(camModelDict[model], width, height, params, dbCameraId)
        updates.append((dbCameraId, camModelDict[model], width, height, params))

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    for dbCameraId, model, width, height, params in updates:
        row = next(db.execute(
            "SELECT model, width, height, params FROM cameras WHERE camera_id=?",
            (dbCameraId,)))
        assert row[0] == model and row[1] == width and row[2] == height
        assert np.allclose(blob_to_array(row[3], np.float64), params)

    # Close database.db.
    db.close()

    with open(args.txt_path, "w") as camFile, open(images_txt, "w") as imgFile:
        for imageId, name, dbCameraId in sorted(dbImages):
            if name not in images:
                continue
            model, width, height, params = cameras[images[name][1]]
            print(dbCameraId, model, width, height, " ".join(params), file=camFile)
            print(imageId, " ".join(images[name][0]), dbCameraId, name, "\n",
                  file=imgFile)
    print("synced %d cameras between %s and the database"
          % (len(updates), args.txt_path))

if __name__ == "__main__":
    import sys,os

    camTodatabase()

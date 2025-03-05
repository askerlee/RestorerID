import os
# Suppress tensorflow info and warning messages. This should be before importing deepface.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_USE_LEGACY_KERAS"] = '1'

import re
import glob
import numpy as np
from PIL import Image
import cv2
import time

from adaface.util import pad_image_obj_to_square
import traceback

def set_tf_gpu(gpu_id):
    import tensorflow as tf
    if gpu_id >= 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            except RuntimeError as e:
                print(e)
    else:
        cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpus, device_type='CPU')
        tf.config.experimental.set_visible_devices([], 'GPU')

def expand_paths(paths, filter_pat="_mask.png", num_samples=-1):
    img_extensions = [ "jpg", "jpeg", "png", "bmp" ]

    if not isinstance(paths, (list, tuple)):
        paths = [ paths ]

    all_paths = []
    for path in paths:
        if os.path.isfile(path):
            all_paths.append(path)
        else:
            for ext in img_extensions:
                all_paths += glob.glob(path + "/*" + ext)

    # Remove mask images.
    all_paths = filter(lambda x: not x.endswith(filter_pat), all_paths)
    all_paths = sorted(all_paths)
    if num_samples > 0:
        all_paths = all_paths[-num_samples:]
    return all_paths

# MonkeyPatch_RetinaFace_Pytorch: Monkey patch deepface.models.face_detection.RetinaFace 
# with the pytorch counterpart. The original RetinaFace is in tensorflow, which is very slow.
# The pytorch implementation is slightly faster.
def deepface_embed_images(image_paths, model_name='ArcFace', detector_backend='retinaface', 
                          enforce_detection=True, align=True, normalization="base", size=(512, 512),
                          cache_embeds=False, MonkeyPatch_RetinaFace_Pytorch=True):
    """
    This function extracts faces from a list of images, and embeds them as embeddings. 

    Parameters:
            image_paths: exact image paths as a list of strings. numpy array (BGR) or based64 encoded
            images are also welcome. If one of pair has more than one face, then we will compare the
            face pair with max similarity.

            model_name (str): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib
            , ArcFace and SFace

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib or mediapipe

    Returns:
            Returns a list of embeddings.

    """
    from deepface import DeepFace

    if MonkeyPatch_RetinaFace_Pytorch:
        import sys
        from evaluation import retinaface_pytorch
        import deepface
        import deepface.modules.modeling
        # Monkey patch deepface.models.face_detection.RetinaFace with the pytorch version.
        # The original RetinaFace is in tensorflow, which is very slow.
        # NOTE: RetinaFace is only for face detection, not for embedding, which is done in DeepFace.represent().
        sys.modules['deepface.models.face_detection.RetinaFace'] = retinaface_pytorch
        tasks = ['facial_recognition', 'spoofing', 'facial_attribute', 'face_detector']
        if 'cached_models' not in deepface.modules.modeling.__dict__:
            deepface.modules.modeling.cached_models = { task: {} for task in tasks }

        # Replace the original tensorflow retinaface with the pytorch version, which is much faster.
        deepface.modules.modeling.cached_models['face_detector']['retinaface'] = retinaface_pytorch.RetinaFaceClient()

    # --------------------------------
    all_embeddings = []
    det_time = 0
    rep_time = 0
    global cached_embeddings

    for img_path in image_paths:
        if not "cached_embeddings" in globals():
            cached_embeddings = {}
        if img_path in cached_embeddings:
            embeddings = cached_embeddings[img_path]
            all_embeddings.append(embeddings)
            continue

        embeddings = []

        image_obj = Image.open(img_path)
        image_obj2, _, _ = pad_image_obj_to_square(image_obj)
        # Resize image to (512, 512). The scheme is Image.NEAREST, to be consistent with 
        # PersonalizedBase dataset class.
        image_obj2 = image_obj2.resize(size, Image.NEAREST)
        # Keep the original RGB image for face detection.
        image_np = np.array(image_obj2)

        try:
            start = time.time()
            # img_path might have many faces
            img_objs = DeepFace.extract_faces(
                img_path=image_np,
                detector_backend=detector_backend,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
            )

            det_time += time.time() - start

        except Exception as e: 
            print(img_path)
            traceback.print_exc()
            continue
        
        start = time.time()
        # now we will find the face pair with minimum distance
        for img_obj in img_objs:
            img_content = img_obj["face"]
            img_embedding_obj = DeepFace.represent(
                img_path=img_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            embedding = img_embedding_obj[0]["embedding"]
            embeddings.append(embedding)
        rep_time += time.time() - start
        #print(f"det_time: {det_time:.3f}, rep_time: {rep_time:.3f}")
        embeddings = np.array(embeddings)
        all_embeddings.append(embeddings)
        if cache_embeds:
            cached_embeddings[img_path] = embeddings

    return all_embeddings

def insightface_embed_images(insightface_app, image_paths, gpu_id=0, size=(512, 512)):
    """
    This function extracts faces from a list of images, and embeds them as embeddings. 

    Parameters:
            image_paths: exact image paths as a list of strings. numpy array (BGR) or based64 encoded
            images are also welcome. If one of pair has more than one face, then we will compare the
            face pair with max similarity.

            model_name (str): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib
            , ArcFace and SFace

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib or mediapipe

    Returns:
            Returns a list of embeddings.

    """

    # Only for one-off call. Otherwise it will be very slow.
    if insightface_app is None:
        from insightface.app import FaceAnalysis
        # FaceAnalysis will try to find the ckpt in: models/insightface/models/antelopev2. 
        # Note there's a second "model" in the path.        
        insightface_app = FaceAnalysis(name='antelopev2', root='models/insightface', providers=['CPUExecutionProvider'])
        insightface_app.prepare(ctx_id=gpu_id, det_size=(512, 512))

    image_nps = []
    for image_path in image_paths:
        image_obj = Image.open(image_path)
        image_obj2, _, _ = pad_image_obj_to_square(image_obj)
        # Resize image to (512, 512). The scheme is Image.NEAREST, to be consistent with 
        # PersonalizedBase dataset class.
        image_obj2 = image_obj2.resize(size, Image.NEAREST)
        image_np = np.array(image_obj2)
        image_np2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_nps.append(image_np2)

    all_embeddings = []
    for idx, image_np in enumerate(image_nps):
        face_infos = insightface_app.get(image_np)
        if len(face_infos) > 0:
            face_info = sorted(face_infos, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
            # face_info.normed_embedding: [512,]. Already np.array, so no need to convert.
            all_embeddings.append([face_info.normed_embedding])
        else:
            # len(face_info) == 0 and skip_non_faces.
            # Skip images without faces.
            print(f'Skip image without face: {image_paths[idx]}')
            # Append an empty list to indicate that there is no face in this image.
            all_embeddings.append([])

    return all_embeddings


# src_embeds: N1 * embed_dim, dst_embeds: N2 * embed_dim
# return a similarity matrix of N1 * N2.
def np_cosine_similarity(src_embeds, dst_embeds):
    a = np.matmul(src_embeds, np.transpose(dst_embeds))
    b = np.sum(np.multiply(src_embeds, src_embeds), axis=1, keepdims=True)
    c = np.sum(np.multiply(dst_embeds, dst_embeds), axis=1, keepdims=True)
    return (a / (np.sqrt(b) * np.sqrt(c).T))
    
def calc_faces_mean_similarity(src_list_embeds, dst_list_embeds):
    """
    This function calculates similarity between two lists of face embeddings.

    Parameters:
            src_list_embeds: list of embeddings as numpy array
            dst_list_embeds: list of embeddings as numpy array

    Returns:
            Returns a list of similarity scores.
    """
    # --------------------------------
    # now we will find the face pair with minimum distance
    all_similarities = []
    src_no_face_img_count = 0
    dst_no_face_img_count = 0

    for src_embeds in src_list_embeds:
        if len(src_embeds) == 0:
            src_no_face_img_count += 1
    for dst_embeds in dst_list_embeds:
        if len(dst_embeds) == 0:
            dst_no_face_img_count += 1

    for src_embeds in src_list_embeds:
        if len(src_embeds) == 0:
            continue
        for dst_embeds in dst_list_embeds:
            if len(dst_embeds) == 0:
                continue
            # src_embeds: (num_faces1, embed_dim)
            # dst_embeds: (num_faces2, embed_dim)
            # sim_matrix: (num_faces1, num_faces2)
            sim_matrix = np_cosine_similarity(src_embeds, dst_embeds)
            max_sim = np.max(sim_matrix)
            all_similarities.append(max_sim)

    if len(all_similarities) == 0:
        mean_similarity = 0
    else:
        mean_similarity = np.mean(all_similarities)

    return mean_similarity, src_no_face_img_count, dst_no_face_img_count

# src_path, dst_path: a folder or a single image path
def compare_face_folders(src_path, dst_path, src_num_samples=-1, dst_num_samples=-1, 
                         face_engine="deepface", insightface_app=None, 
                         cache_src_embeds=True, verbose=True):

    src_paths = expand_paths(src_path, num_samples=src_num_samples)
    dst_paths = expand_paths(dst_path, num_samples=dst_num_samples)

    if face_engine == "deepface":
        src_list_embeds = deepface_embed_images(src_paths, model_name="ArcFace", detector_backend = "retinaface",
                                                cache_embeds=cache_src_embeds)
        dst_list_embeds = deepface_embed_images(dst_paths, model_name="ArcFace", detector_backend = "retinaface",
                                                cache_embeds=False)
    elif face_engine == "insightface":
        src_list_embeds = insightface_embed_images(insightface_app, src_paths)
        dst_list_embeds = insightface_embed_images(insightface_app, dst_paths)
    else:
        breakpoint()

    '''
    if face_engine == "deepface":
        (Pdb) calc_faces_mean_similarity(src_list_embeds, dst_list_embeds)
        (0.471041, 0, 0)
        (Pdb) calc_faces_mean_similarity(src_list_embeds, src_list_embeds)
        (0.622069, 0, 0)
        (Pdb) calc_faces_mean_similarity(dst_list_embeds, dst_list_embeds)
        (0.660250, 0, 0)
    if face_engine == "insightface":
        (Pdb) calc_faces_mean_similarity(src_list_embeds, dst_list_embeds)
        (0.339248, 0, 0)
        (Pdb) calc_faces_mean_similarity(src_list_embeds, src_list_embeds)
        (0.689450, 0, 0)
        (Pdb) calc_faces_mean_similarity(dst_list_embeds, dst_list_embeds)
        (0.480570, 0, 0)
    Seems that insightface embeddings are very sensitive to details like lightning, pose and tone.
    Therefore, by default we use deepface embeddings, as they only focus on face characteristics.
    '''

    avg_similarity, src_no_face_img_count, dst_no_face_img_count =\
        calc_faces_mean_similarity(src_list_embeds, dst_list_embeds)
    
    dst_normal_img_count = len(dst_paths) - dst_no_face_img_count

    if isinstance(src_path, (list, tuple)):
        src_path = src_path[0]
    if isinstance(dst_path, (list, tuple)):
        dst_path = dst_path[0]

    if src_path[-1] == "/":
        src_path = src_path[:-1]
    if dst_path[-1] == "/":
        dst_path = dst_path[:-1]
    src_path_base = os.path.basename(src_path)
    dst_path_base = os.path.basename(dst_path)
    if verbose:
        print(f"avg face sim: {avg_similarity:.3f}    '{src_path_base}' vs '{dst_path_base}' ({dst_no_face_img_count} no face)")
        
    return avg_similarity, dst_normal_img_count, dst_no_face_img_count

# extra_sig could be a regular expression
def find_first_match(lst, search_term, extra_sig=""):
    for item in lst:
        if search_term in item and re.search(extra_sig, item):
            return item
    return None  # If no match is found

# if fix_1_offset: range_str "3-7,8,10" => [2, 3, 4, 5, 6, 7, 9]
# else:            range_str "3-7,8,10" => [3, 4, 5, 6, 7, 8, 10]
# "a-b" is always inclusive, i.e., "a-b" = [a, a+1, ..., b]
def parse_range_str(range_str, fix_1_offset=False):
    if range_str is None:
        return None
    
    result = []
    offset = 1 if fix_1_offset else 0

    for part in range_str.split(','):
        if '-' in part:
            a, b = part.split('-')
            a, b = int(a) - offset, int(b) - offset
            result.extend(list(range(a, b + 1)))
        else:
            a = int(part) - offset
            result.append(a)
    return result

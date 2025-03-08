from basicsr.metrics import calculate_psnr, calculate_ssim
from evaluation.eval_utils import compare_face_folders
import sys, os, cv2

if len(sys.argv) != 3:
    print('Usage: python calc_metrics.py <samples_folder> <results_folder>')
    sys.exit(1)

samples_folder = sys.argv[1]
results_folder = sys.argv[2]

methods = [ 'adaface', 'ipadapter', 'consistentID', 'arc2face', 'consistentID-arc2face' ]
stats   = { 'psnr': {}, 'ssim': {}, 'face_sim': {} }
face_engine = 'deepface'
    
# subj_folder: 1, 10, 2, ..., 9
for subj_folder in sorted(os.listdir(samples_folder)):
    lq1_gt_path = os.path.join(samples_folder, subj_folder, 'lq1-gt.png')
    lq1_path    = os.path.join(samples_folder, subj_folder, 'lq1.png')
    ref1_path   = os.path.join(samples_folder, subj_folder, 'ref1.png')

    subj_results_folder = os.path.join(results_folder, subj_folder)
    gt_img_np = cv2.imread(lq1_gt_path, cv2.IMREAD_UNCHANGED)

    for restored_img_filename in sorted(os.listdir(subj_results_folder)):
        method = '-'.join(restored_img_filename.split('-')[1:])[:-4]
        if not method in methods:
            breakpoint()

        restored_img_path = os.path.join(subj_results_folder, restored_img_filename)
        restored_img_np = cv2.imread(restored_img_path, cv2.IMREAD_UNCHANGED)

        psnr = calculate_psnr(gt_img_np, restored_img_np, crop_border=4, 
                              input_order='HWC', test_y_channel=False)
        ssim = calculate_ssim(gt_img_np, restored_img_np, crop_border=4, 
                              input_order='HWC', test_y_channel=False)
        stats['psnr'].setdefault(method, []).append(psnr)
        stats['ssim'].setdefault(method, []).append(ssim)

        face_sim, _, _ = compare_face_folders([lq1_gt_path], [restored_img_path], 
                                              face_engine=face_engine, verbose=False)
        stats['face_sim'].setdefault(method, []).append(face_sim)
        print(f'{subj_folder}-{method}\tPSNR: {psnr:.3f}  SSIM: {ssim:.3f}  FaceSim: {face_sim:.3f}')

avg_stats = {}
for method in methods:
    if method in stats['psnr']:
        for metric in ['psnr', 'ssim', 'face_sim']:
            #stats[metric][method] = sum(stats[metric][method]) / len(stats[metric][method])
            avg_stats.setdefault(method, {})[metric] = \
                sum(stats[metric][method]) / len(stats[metric][method])

print('\nAverage metrics:')
for method in avg_stats:
    print(f'{method}\tPSNR: {avg_stats[method]["psnr"]:.3f}  SSIM: {avg_stats[method]["ssim"]:.3f}  FaceSim: {avg_stats[method]["face_sim"]:.3f}')


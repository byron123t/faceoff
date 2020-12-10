import Config
import os
import imageio
import numpy as np

from deepface import DeepFace
from scaleatt.scaling.ScalingGenerator import ScalingGenerator
from scaleatt.scaling.SuppScalingLibraries import SuppScalingLibraries
from scaleatt.scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaleatt.scaling.ScalingApproach import ScalingApproach
from scaleatt.attack.QuadrScaleAttack import QuadraticScaleAttack
from scaleatt.attack.ScaleAttackStrategy import ScaleAttackStrategy

from scaleatt.defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from scaleatt.defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator
from scaleatt.defenses.prevention.PreventionDefense import PreventionDefense

from scaleatt.defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector, PeakMatrixMethod
from scaleatt.attack.adaptive_attack.AdaptiveAttackPreventionGenerator import AdaptiveAttackPreventionGenerator
from scaleatt.attack.adaptive_attack.AdaptiveAttack import AdaptiveAttackOnAttackImage


BACKENDS = [
                # 'opencv',
                'mtcnn',
                # 'ssd'
           ]
SHAPES = {
            'deepid':(55, 47),
            'deepid1':(64, 64),
            'openface':(96, 96),
            'openface1':(100, 100),
            'openface2':(120, 120),
            'openface3':(144, 144),
            'deepface':(152, 152),
            'deepface1':(154, 154),
            'deepface2':(156, 156),
            'deepface3':(158, 158),
            'facenet':(160, 160),
            'facenet1':(162, 162),
            'facenet2':(164, 164),
            # 'vggface':(224, 224),
         }
PATH = os.path.join(Config.ROOT, 'data')
ALGORITHMS = {
                 # 'nearest':SuppScalingAlgorithms.NEAREST,
                 # 'linear':SuppScalingAlgorithms.LINEAR,
                 'cubic':SuppScalingAlgorithms.CUBIC,
                 'lanczos':SuppScalingAlgorithms.LANCZOS,
                 # 'area':SuppScalingAlgorithms.AREA
             }
SOURCE = 'meryl'
TARGET = 'matt'
args_bandwidthfactor: int = 2
usecythonifavailable: bool = True
for backend in BACKENDS:
    for shape_key, face_shape in SHAPES.items():
        for algo_key, algo in ALGORITHMS.items():
            try:
                print(backend, shape_key, algo_key)
                src_file = os.path.join(PATH, 'test_imgs', SOURCE, '{}_01.jpg'.format(SOURCE))
                trg_file = os.path.join(PATH, 'test_imgs', TARGET, '{}_01.jpg'.format(TARGET))
                #face detection and alignment
                src_align, src_bounds, src_cropped, src_img = DeepFace.detectFace(src_file, detector_backend = backend, return_bounds = True, target_size = face_shape)
                trg_align, trg_bounds, trg_cropped, trg_img = DeepFace.detectFace(trg_file, detector_backend = backend, return_bounds = True, target_size = face_shape)

                scaling_algorithm: SuppScalingAlgorithms = algo
                scaling_library: SuppScalingLibraries = SuppScalingLibraries.CV
                scaler_approach: ScalingApproach = ScalingGenerator.create_scaling_approach(
                    x_val_source_shape=src_cropped.shape,
                    x_val_target_shape=trg_align.shape,
                    lib=scaling_library,
                    alg=scaling_algorithm
                )
                scale_att: ScaleAttackStrategy = QuadraticScaleAttack(eps=1, verbose=True)

                src_cropped = (src_cropped * 255).astype(np.uint8)
                trg_align = (trg_align * 255).astype(np.uint8)
                src_img = (src_img * 255).astype(np.uint8)
                print(src_cropped.shape)
                print(trg_align.shape)

                result_attack_image, _, _ = scale_att.attack(src_image=src_cropped,
                                                             target_image=trg_align,
                                                             scaler_approach=scaler_approach)
                fourierpeakmatrixcollector: FourierPeakMatrixCollector = FourierPeakMatrixCollector(
                    method=PeakMatrixMethod.optimization, scale_library=scaling_library, scale_algorithm=scaling_algorithm
                )
                args_prevention_type = PreventionTypeDefense.medianfiltering
                preventiondefense: PreventionDefense = PreventionDefenseGenerator.create_prevention_defense(
                            defense_type=args_prevention_type, scaler_approach=scaler_approach,
                            fourierpeakmatrixcollector=fourierpeakmatrixcollector,
                            bandwidth=args_bandwidthfactor, verbose_flag=False, usecythonifavailable=usecythonifavailable
                        )


                ## New! Init and run the adaptive attack
                args_allowed_changes: int = 50 # the percentage of pixels that can be modified in each block.
                adaptiveattack: AdaptiveAttackOnAttackImage = AdaptiveAttackPreventionGenerator.create_adaptive_attack(
                            defense_type=args_prevention_type, scaler_approach=scaler_approach,
                            preventiondefense=preventiondefense,
                            verbose_flag=False, usecythonifavailable=usecythonifavailable,
                            choose_only_unused_pixels_in_overlapping_case=False,
                            allowed_changes=args_allowed_changes/100
                        )

                adaptiveattackimage = adaptiveattack.counter_attack(att_image=result_attack_image)
                # Now we use the attack image and scale it down, as we would do in a real machine learning pipeline.
                result_output_image = scaler_approach.scale_image(xin=adaptiveattackimage)
                src_img[src_bounds[0]:src_bounds[1], src_bounds[2]:src_bounds[3]] = adaptiveattackimage
                imageio.imwrite(os.path.join(PATH, 'output_scaled', '{}_{}_{}_{}_{}_01.jpg'.format(backend, shape_key, algo_key, SOURCE, TARGET)), src_img)
            except Exception as e:
                print(e)
                continue
    #face verification
    #obj = DeepFace.verify("img1.jpg", "img2.jpg", detector_backend = backend)

    #face recognition
    #df = DeepFace.find(img_path = "img.jpg", db_path = "my_db", detector_backend = backend)

    #facial analysis
    #demography = DeepFace.analyze("img4.jpg", detector_backend = backend)

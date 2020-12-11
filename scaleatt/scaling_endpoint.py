import os
import imageio
import numpy as np

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

ALGORITHMS = {'nearest':SuppScalingAlgorithms.NEAREST,
              'bilinear':SuppScalingAlgorithms.LINEAR,
              'bicubic':SuppScalingAlgorithms.CUBIC,
              'lanczos':SuppScalingAlgorithms.LANCZOS,
              'area':SuppScalingAlgorithms.AREA}
args_bandwidthfactor: int = 2
usecythonifavailable: bool = True
args_allowed_changes: int = 20 # the percentage of pixels that can be modified in each block.


def scale_attack(delta, src_img):
    print(src_img.shape)
    print(delta.shape)
    #src_img = np.zeros((orig_dim[0], orig_dim[1], 3), dtype=np.uint8)
    scaling_algorithm: SuppScalingAlgorithms = ALGORITHMS['bilinear']
    scaling_library: SuppScalingLibraries = SuppScalingLibraries.CV
    scaler_approach: ScalingApproach = ScalingGenerator.create_scaling_approach(
        x_val_source_shape=src_img.shape,
        x_val_target_shape=delta.shape,
        lib=scaling_library,
        alg=scaling_algorithm
    )
    scale_att: ScaleAttackStrategy = QuadraticScaleAttack(eps=1, verbose=True)

    result_attack_image, _, _ = scale_att.attack(src_image=src_img,
                                                 target_image=delta,
                                                 scaler_approach=scaler_approach)
    
    # fourierpeakmatrixcollector: FourierPeakMatrixCollector = FourierPeakMatrixCollector(
    #     method=PeakMatrixMethod.optimization, scale_library=scaling_library, scale_algorithm=scaling_algorithm)

    # args_prevention_type = PreventionTypeDefense.medianfiltering
    # preventiondefense: PreventionDefense = PreventionDefenseGenerator.create_prevention_defense(
    #             defense_type=args_prevention_type, scaler_approach=scaler_approach,
    #             fourierpeakmatrixcollector=fourierpeakmatrixcollector,
    #             bandwidth=args_bandwidthfactor, verbose_flag=False, usecythonifavailable=usecythonifavailable)

    # ## New! Init and run the adaptive attack
    # adaptiveattack: AdaptiveAttackOnAttackImage = AdaptiveAttackPreventionGenerator.create_adaptive_attack(
    #             defense_type=args_prevention_type, scaler_approach=scaler_approach,
    #             preventiondefense=preventiondefense,
    #             verbose_flag=False, usecythonifavailable=usecythonifavailable,
    #             choose_only_unused_pixels_in_overlapping_case=False,
    #             allowed_changes=args_allowed_changes/100)

    # adaptiveattackimage = adaptiveattack.counter_attack(att_image=result_attack_image)

    return result_attack_image

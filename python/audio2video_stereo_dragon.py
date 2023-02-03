# -*- coding: utf-8 -*-
"""
@author: YASUHARA Wataru

MIT License

"""

from analysis import LoadWav
from analysis import wave_analysis
from visualization import make_frame_stereo_dragon as visualize

def main():
    wave_data_file_name = r"test_stereo_diff.wav"
    wave_data_folder = r"test_data\\"
    img_output_folder = r"python\\temp_file\\"
    fps = 30
    au_display_sample_num = int(44100*0.8)
    delay_time = 0.5
    display_freq_max = 1500
    output_width = 160
    output_height = 120
    file_num_zfill = 10
    
    au_data,au_data_time,a_sr = LoadWav.load_wav(wave_data_folder+wave_data_file_name)
    
    ch1_wave_params = wave_analysis.calc_wave_params(au_data[0],
                                    au_data_time,
                                    a_sr,fps,
                                    au_display_sample_num)  
    ch2_wave_params = wave_analysis.calc_wave_params(au_data[1],
                                    au_data_time,
                                    a_sr,fps,
                                    au_display_sample_num)  

    visualize.make_frame_cir_stereo_smooth(ch1_wave_params,
                                            ch2_wave_params,
                                            img_output_folder,
                                            fps,
                                            au_display_sample_num,
                                            display_freq_max,
                                            delay_time,
                                            output_width,
                                            output_height,
                                            file_num_zfill)

    
if __name__ == "__main__":   
    main()

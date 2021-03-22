import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import csv
import pickle
import random
import time

import numpy as np
import tensorflow as tf

import cv2
from collections import Counter
from constants import *
from tensorflow.keras.models import load_model
from video_stream import FileVideoStream

print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


class LakituAI:

    def __init__(self, video_path, gamecoords, xcam_coords, reset_coords, coin_coords,
        star_coords, vod_id, vod_date, speedrunner, run_type, vod_source, current_frame=0, framerate=29.97):
        self.framerate = framerate
        self.video = FileVideoStream(video_path, current_frame).start()
        # self.video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        self.current_frame_image = self.video.read()
        self.vod_current_frame = current_frame
        self.final_frame = self.video.final_frame()
        self.vod_source = vod_source
        self.run_type = run_type
        
        self.vod_date = vod_date
        self.speedrunner = speedrunner

        self.vod_id = vod_id
        self.gamecoords = gamecoords
        self.xcam_coords = xcam_coords
        self.reset_coords = reset_coords
        self.coin_coords = coin_coords
        self.star_coords = star_coords

        self.xcam_model = load_model(XCAMMODEL_PATH)
        self.check_xcam() # run model to set it up

        self.frame_model = load_model(MODEL_PATH)
        self.label_keys = list(pickle.load(open(LABEL_DICT_PATH, "rb")).keys())
        self.check_model() # run model to set it up

        self.coin_model = load_model(COINMODEL_PATH)
        self.check_coins()

        self.star_model = load_model(STARMODEL_PATH)

        self.classification_poll = []

    def watch(self):
        self.status = 'waiting_for_reset'
        self.level = 'None'
        self.show_new_record = 0
        previous_frame = None
        self.this_run_frames = 0
        self.clutch = False
        self.last_fade_run = 0
        self.last_fade_vod = 0
        self.vod_start = 0
        self.start_time = 0
        self.acquired_stars = []
        self.star_count = 0
        self.run_number = 1
        self.xcam_frame = 0
        self.timelog = [0 for _ in range(30)]
        old_status = None
        self.care_xcam = False
        xcam_value = True
        self.pause_xcam = False
        self.fps = 0
        self.previous_level = 'run_start'
        self.movement_start = 0
        self.movement_framestamp = 0
        self.show_time = 0
        self.record_added = False

        # # override
        # self.status = 'waiting_for_first_xcam'
        # # self.drop_clutch('level_start')
        # self.care_xcam = True
        # self.star_count = 62
        # self.previous_level = 'run_start'
        # self.star_start = 0
        # self.acquired_stars = ['castle__mips_1']
        # self.framestamp = 0
        # self.run_number = 17

        while self.video.more():
            start_time = time.time()

            # self.current_frame_classification = self.classify_frame()
            self.process_frame()
            fade_value = self.check_fade()

            if self.care_xcam:
                xcam_value = self.check_xcam()
                if self.pause_xcam:
                    if xcam_value in ['mariocam', 'lakitucam', 'blank']:
                            self.pause_xcam = False
                    else:
                        xcam_value = None

            if old_status != self.status:
                print(f'current status: {self.status}')
                old_status = self.status
                if self.clutch:
                    print('clutch is dropped')

            # catch fades
            if fade_value in FADES:
                self.last_fade_run = self.this_run_frames
                self.last_fade_vod = self.vod_current_frame

            # Detect a reset
            reset_val = self.check_reset()
            
            if reset_val > 0:
                print("reset detected")
                if self.status == 'waiting_for_star':
                    self.add_record('None', event_type = 'incomplete')

                if self.record_added:
                    self.run_number += 1
                self.release_clutch()
                self.level = 'None'
                self.this_run_frames = 39 + reset_val
                # beginning_blackfades = 3
                self.acquired_stars = []
                self.star_count = 0
                self.movement_start = 0
                self.movement_framestamp = self.vod_current_frame - (39 + reset_val)
                self.status = 'waiting_for_first_xcam'
                self.previous_level = 'run_start'
                self.care_xcam = True
                self.record_added = False

            if self.status == 'waiting_for_first_xcam':
                if xcam_value == 'xcam':
                    self.status = 'waiting_for_castle_entry'

            # ZOOOOOM through intro sequence
            if self.status == 'waiting_for_castle_entry':

                if previous_frame == 'black_fade' and \
                  fade_value != 'black_fade':
                    self.status = 'waiting_for_trigger'

            if self.status == 'waiting_for_trigger':
                
                if fade_value in FADES and xcam_value == 'blank':
                    if not self.clutch:
                        print('fade detected for trigger, dropping clutch')   
                        previous_frame = None         
                        self.drop_clutch('triggers')

                elif xcam_value == 'xcam':
                    
                    if self.check_stars() == (self.star_count + 1):
                        if not self.clutch:
                            print('star increase detected, checking for event star, dropping clutch')
                            self.drop_clutch('event_stars')

            if self.status == 'waiting_for_level_determination':

                if previous_frame == 'white_fade' and fade_value != 'white_fade':
                    if not self.clutch:
                        print('white fade detected for level determination, dropping clutch')
                        self.drop_clutch('level_start')

            if self.status == 'waiting_for_star':

                if xcam_value == 'xcam':
                    if not self.clutch:
                        if self.level in BOWSER_LEVELS and \
                          str(self.level + "__key") not in self.acquired_stars:
                            print('xcam detected, dropping clutch')
                            self.xcam_frame = self.this_run_frames
                            self.drop_clutch('star')
                        
                        elif self.check_stars() == (self.star_count + 1):
                            self.drop_clutch('star')
                            print('star increase detected, dropping clutch')

                elif xcam_value == 'blank' and fade_value in FADES:
                    if not self.clutch:
                        print('fade detected, checking if trigger')
                        self.drop_clutch('possible_trigger')

            if self.status == 'waiting_for_determination_err':

                if previous_frame == 'white_fade' and fade_value != 'white_fade':
                    if not self.clutch:
                        print('white fade detected for level determination, dropping clutch')
                        self.drop_clutch('error_determination')

                # elif xcam_value in ['mariocam', 'lakitucam']:
                #     if self.clutch:
                #         print('non-xcam detected, releasing clutch')
                #         self.release_clutch()

            if self.status == 'waiting_for_fade_clear':
                if self.check_xcam() != 'blank':
                    self.pause_xcam = True
                    self.status = 'waiting_for_trigger'
                    self.movement_start = self.last_fade_run
                    self.movement_framestamp = self.last_fade_vod

            previous_frame = fade_value
            self.display_frame()
            self.advance_frame()

            self.timelog.pop(0)
            self.timelog.append(time.time() - start_time)
            self.fps = int( 1 / (sum(self.timelog)/len(self.timelog)))

    def add_record(self, label, event_type='star'):

        self.record_added = True
        self.event_type = event_type
        
        if label is None or label == 'None':
            if event_type == 'incomplete':
                self.show_label = 'incomplete'
            elif event_type == 'star':
                self.show_label = 'unknown'
                label = str(self.level+'__unknown')
        else:
            self.show_label = label

        # id, runner, date, framerate, run number, event type, vod framestamp
        record = [self.vod_id, self.speedrunner, self.run_type, self.vod_source, 
            self.vod_date, self.framerate, self.run_number, event_type]

        if event_type == 'star':
            # Don't increase star count for keys
            if label not in NON_STARS:
                self.star_count += 1

            # If the star is a castle event star, use movement as the start time
            if label in EVENT_STARS:
                self.level = 'castle'
                start_frame = self.movement_start
                framestamp = self.movement_framestamp
            else:
                start_frame = self.star_start
                framestamp = self.framestamp

            # If the star is a continue or event star, use the first analyzed frame as the end time
            if label in CONTINUE_STARS or label in EVENT_STARS:
                end_frame = self.frame_clutch_dropped
            else:
                end_frame = self.this_run_frames

            # Add the star to the acquired star list
            self.acquired_stars.append(label)
            
            # Build and add the record
            addon_fields = [framestamp, self.level, label.split('__')[1], self.star_count,
                start_frame, end_frame]
            record = record + addon_fields
            self.show_time = (end_frame - start_frame) / self.framerate

        elif event_type == 'castle_movement':
            # level from, level to, star count, run start frame, run end frame
            addon_fields = [self.movement_framestamp, label.split('__')[0], label.split('__')[1], self.star_count, 
                self.movement_start, self.last_fade_run]
            record = record + addon_fields
            self.show_time = (self.last_fade_run - self.movement_start) / self.framerate
            
        elif event_type == 'incomplete':
            addon_fields = [self.framestamp, self.level, 'unknown', self.star_count,
                self.star_start, self.this_run_frames]
            record = record + addon_fields
            self.show_time = (self.this_run_frames - self.star_start) / self.framerate

        with open(f'outputs\\{self.vod_id}.csv', 'a', newline='') as f:
            csv_writer = csv.writer(f) 
            csv_writer.writerow(record)
        
        self.show_new_record = int(self.framerate * 8)
        print(f'{label} ({self.star_count})')       

    def check_coins(self):
        
        feature = self.current_frame_image[
            self.coin_coords[0] : self.coin_coords[1],
            self.coin_coords[2] : self.coin_coords[3]
        ]
        
        feature = cv2.resize(feature, (50,50), interpolation=cv2.INTER_LINEAR)
        feature = np.asarray([feature])
        feature = feature/255.0

        coincheck = np.argmax(self.coin_model(feature, training=False))
        if coincheck == 0:
            return True
        else:
            return False

    def check_stars(self):
        
        feature = self.current_frame_image[
            self.star_coords[0] : self.star_coords[1],
            self.star_coords[2] : self.star_coords[3]
        ]
        
        feature = cv2.cvtColor(feature, cv2.COLOR_BGR2RGB)

        feature = cv2.resize(feature, (98,98), interpolation=cv2.INTER_LINEAR)
        feature = np.asarray([feature])
        # feature = feature/255.0

        output = self.star_model(feature, training=False)
        starcheck = np.argmax(output)
        certainty = output[0][starcheck]
        # print(starcheck, certainty)
        if certainty >= 0.9:
            if starcheck == 121:
                return None
            else:
                return starcheck
        else:
            return None

    def process_frame(self):

        if self.clutch:
            self.clutch_duration += 1
            label, certainty = self.check_model()

            if self.status == 'waiting_for_trigger':
                if self.clutch_context == 'triggers':
                    if self.clutch_duration >= (self.framerate * 3):
                        self.release_clutch()
                    elif label in CASTLE_STARTS or label == 'misc__star_select':
                        self.classification_poll.append([label, certainty])
                        poll_winner, votes = self.count_poll()
                        if votes >= int(self.framerate * 0.33):
                            if poll_winner == 'misc__star_select':
                                self.status = 'waiting_for_level_determination'
                            else:
                                self.status = 'waiting_for_star'
                                self.star_start = self.last_fade_run
                                self.framestamp = self.last_fade_vod
                                self.level = poll_winner.split('__')[1]
                                self.care_xcam = True
                                print(self.level)
                                if self.level != self.previous_level:
                                    self.add_record(
                                        str(self.previous_level + '__' + self.level),
                                        'castle_movement'
                                    )
                                self.previous_level = self.level
                            self.release_clutch()

                elif self.clutch_context == 'event_stars':

                    if self.check_fade() in ['black_fade', 'white_fade']:
                        self.release_clutch()
                    elif self.check_xcam() in ['mariocam', 'lakitucam']:
                        self.release_clutch()

                    if label == 'castle__mips':
                        if 'castle__mips_1' not in self.acquired_stars and \
                          self.star_count >= 15:
                            label = 'castle__mips_1'
                            self.classification_poll.append([label, certainty])
                        elif 'castle__mips_2' not in self.acquired_stars and \
                          self.star_count >= 50:
                            label = 'castle__mips_2'
                            self.classification_poll.append([label, certainty])
                    elif label in TOAD_STARS:
                        if label not in self.acquired_stars:
                            self.classification_poll.append([label, certainty])

                    poll_winner, votes = self.count_poll()

                    if votes >= int(self.framerate * 0.33) or self.clutch_duration >= int(self.framerate * 2):
                        self.add_record(poll_winner)
                        self.release_clutch()

            elif self.status == 'waiting_for_level_determination':
                if label.split('__')[0] == 'level_start':
                    self.classification_poll.append([label, certainty])
                poll_winner, votes = self.count_poll()
                # print(self.clutch_duration, "/", int(self.framerate * 3))
                if votes >= int(self.framerate * 0.5) or self.clutch_duration >= int(self.framerate * 3):
                    self.status = 'waiting_for_star'
                    self.star_start = self.last_fade_run
                    self.framestamp = self.last_fade_vod
                    self.level = poll_winner.split('__')[1]
                    self.care_xcam = True
                    print(self.level)
                    if self.level != self.previous_level:
                        self.add_record(
                            str(self.previous_level + '__' + self.level),
                            'castle_movement'
                        )
                    self.previous_level = self.level
                    self.release_clutch()

            elif self.status == 'waiting_for_star':
                if self.clutch_context == 'star':
                    if self.check_xcam() in ['mariocam', 'lakitucam']:
                        self.release_clutch()
                    
                    if str(self.level+'__100_coins') not in self.acquired_stars and \
                      self.level in COINSTAR_LEVELS:
                        if self.check_coins():
                            label = str(self.level+'__100_coins')
                            self.add_record(label)
                            self.release_clutch()
                    
                    if label.split('__')[0] == self.level and \
                    label not in self.acquired_stars:
                        self.classification_poll.append([label, certainty])
                    
                    poll_winner, votes = self.count_poll()
                    
                    if poll_winner in CONTINUE_STARS and votes >= int(self.framerate * .33):
                        self.add_record(poll_winner)
                        self.pause_xcam = True
                        self.release_clutch()
                    
                    elif self.check_fade() == 'black_fade' and self.check_xcam() == 'blank':
                        self.add_record(poll_winner)
                        self.status = 'waiting_for_fade_clear'
                        self.level = 'None'
                        self.release_clutch()

                elif self.clutch_context == 'possible_trigger':
                    if self.clutch_duration >= (self.framerate * 3):
                        self.release_clutch()
                    elif label in CASTLE_STARTS or label == 'misc__star_select':
                        self.classification_poll.append([label, certainty])
                        poll_winner, votes = self.count_poll()
                        if votes >= int(self.framerate * 0.75):
                            print('new level trigger detected:', poll_winner, votes)
                            if poll_winner == 'misc__star_select':
                                self.status = 'waiting_for_determination_err'
                            else:
                                self.status = 'waiting_for_star'
                                self.star_start = self.last_fade_run
                                self.level = poll_winner.split('__')[1]
                                if self.level == self.previous_level:
                                    print('adding incomplete record')
                                    self.add_record(None, 'incomplete')
                                self.care_xcam = True
                                print(self.level)
                            self.release_clutch()

            elif self.status == 'waiting_for_determination_err':
                if label.split('__')[0] == 'level_start':
                    self.classification_poll.append([label, certainty])
                poll_winner, votes = self.count_poll()
                
                if votes >= 10 or self.clutch_duration >= int(self.framerate * 3):

                    self.level = poll_winner.split('__')[1]

                    if self.level != self.previous_level:
                        self.add_record(
                            str(self.previous_level + '__' + self.level),
                            'castle_movement'
                        )
                        self.previous_level = self.level
                    elif self.level == self.previous_level:
                        print('adding incomplete level record')
                        self.add_record(None, 'incomplete')

                    self.status = 'waiting_for_star'
                    self.star_start = self.last_fade_run
                    self.framestamp = self.last_fade_vod
                    self.care_xcam = True
                    print(self.level)
                    self.release_clutch()

    def check_xcam(self):

        feature = self.current_frame_image[
            self.xcam_coords[0] : self.xcam_coords[1],
            self.xcam_coords[2] : self.xcam_coords[3]
        ]

        feature = cv2.resize(feature, (50,50), interpolation=cv2.INTER_LINEAR)
        feature = np.asarray([feature])
        feature = feature/255.0

        xcam = np.argmax(self.xcam_model(feature, training=False))
        if xcam == 0:
            return 'xcam'
        elif xcam == 1:
            return 'lakitucam'
        elif xcam == 2:
            return 'mariocam'
        elif xcam == 3:
            return 'blank'

    def check_reset(self):

        feature = self.current_frame_image[
            self.reset_coords[0] : self.reset_coords[1],
            self.reset_coords[2] : self.reset_coords[3]
        ]

        # cv2.imshow('reset', feature)
        # cv2.waitKey(1)
        
        template_shape = RESET_TEMPLATE_1.shape
        feature = cv2.resize(feature, (template_shape[1], template_shape[0]), interpolation = cv2.INTER_LINEAR)

        check = cv2.minMaxLoc(cv2.matchTemplate(feature, RESET_TEMPLATE_1, cv2.TM_SQDIFF_NORMED))[0]
        # print("template 1:",check)
        if check < 0.1:
            return 1
        else:
            check = cv2.minMaxLoc(cv2.matchTemplate(feature, RESET_TEMPLATE_2, cv2.TM_SQDIFF_NORMED))[0]
            # print("template 2:",check)
            if check < 0.2:
                return 2
            else:
                return 0

    def check_fade(self):

        feature = self.current_frame_image[
            int(self.reset_coords[0] * 1.25) : self.reset_coords[1],
            self.reset_coords[2] : self.reset_coords[3]
        ]

        # cv2.imshow('fadearea', feature)
        # cv2.waitKey(1)

        feature = feature / 255.0
        feature = feature.flatten()
        
        if np.sum(feature < 0.1) > len(feature) * 0.9:
            return 'black_fade'
        elif np.sum(feature > 0.8) > len(feature) * 0.99:
            return 'white_fade'
        else:
            return None

    def check_model(self):

        feature = self.current_frame_image[
            self.gamecoords[0] : self.gamecoords[1],
            self.gamecoords[2] : self.gamecoords[3]
        ]

        save_image = feature
        
        feature = cv2.resize(feature, (299, 299), interpolation=cv2.INTER_LINEAR)
        feature = cv2.cvtColor(feature, cv2.COLOR_BGR2RGB)
        feature = np.asarray([feature])

        output = self.frame_model(feature, training=False)
        label = self.label_keys[np.argmax(output)]
        certainty = output[0][np.argmax(output)]

        if self.vod_current_frame % (self.framerate//4) == 0 and label != 'misc__gameplay':
            savepath = f'model_labeled_data\\' + \
                f'{label.split("__")[0]}\\' + \
                f'{label.split("__")[1]}\\' + \
                f'{self.vod_id}_{self.vod_current_frame}_{int(certainty * 1000)}.jpg'
            cv2.imwrite(savepath, save_image)
        return label, certainty

    def count_poll(self):

        if len(self.classification_poll) > 0:
            polldict = {}
            
            for vote in self.classification_poll:
                if vote[0] not in polldict:
                    polldict[vote[0]] = vote[1]
                else:
                    polldict[vote[0]] += vote[1]
            
            maxval = max(polldict.values())
            maxkey = [k for k, v in polldict.items() if v == maxval][0] # could have multiple

            return maxkey, maxval
        
        else:
            return 'None', 0

    def drop_clutch(self, context):
        self.clutch = True
        self.clutch_duration = 0
        self.clutch_context = context
        self.classification_poll = []
        self.frame_clutch_dropped = self.this_run_frames

    def release_clutch(self):
        self.clutch = False
        self.clutch_duration = 0

    def advance_frame(self):
        self.vod_current_frame += 1
        self.this_run_frames += 1
        self.current_frame_image = self.video.read()

    def display_frame(self):
        fontsize = 1.3

        image = self.current_frame_image
        coordslist = [self.gamecoords, self.xcam_coords, self.coin_coords, self.star_coords]
        # coordslist.append(self.reset_coords)

        # Draw a square around view windows
        offset = 2
        for coords in coordslist:
            image = cv2.rectangle(
                image, 
                (coords[2], coords[0]+offset), 
                (coords[3]-offset, coords[1]-offset), 
                (0,0,255), 
                2
            )
            if offset > 0:
                offset = 0

        # Write status data
        print_list = [
            f'Status: {self.status}',
            f'Level: {self.level}'
        ]
        y = int(self.gamecoords[1] * 0.45)
        for line in print_list:
            image = cv2.putText(
                self.current_frame_image,
                line,
                (self.gamecoords[2] + 10, y), 
                cv2.FONT_HERSHEY_PLAIN, 
                fontsize, 
                (0, 0, 255), 
                2, 
                cv2.LINE_AA
            )
            y += 30

        # Write file data
        print_list = [
            f'Speedrunner: {self.speedrunner}',
            f'VOD date: {self.vod_date}'
        ]
        y = int(self.gamecoords[1] * 0.13)
        for line in print_list:
            image = cv2.putText(
                image,
                line,
                (int(self.gamecoords[2]+10), y), 
                cv2.FONT_HERSHEY_PLAIN, 
                fontsize, 
                (0, 0, 255), 
                2, 
                cv2.LINE_AA
            )
            y += 30

        # Write framecount data
        image = cv2.putText(
                    image,
                    f'Frames: {int(self.vod_current_frame)} / {self.final_frame} ({self.fps} fps)',
                    (self.gamecoords[2]+10, int(self.gamecoords[1]*.93)), 
                    cv2.FONT_HERSHEY_PLAIN, 
                    fontsize, 
                    (0, 0, 255), 
                    2, 
                    cv2.LINE_AA
                )

        # Write new record data
        if self.show_new_record > 0:
            image = cv2.rectangle(
                image, 
                (0, int(self.gamecoords[1] * 0.72)), 
                (image.shape[1], int(self.gamecoords[1] * 0.79)), 
                (0,0,0), 
                -1
            )

            image = cv2.putText(
                image,
                f"Adding {self.event_type}: {self.show_label} ({self.show_time:.2f} sec)",
                (10, int(self.gamecoords[1] * 0.77)),
                cv2.FONT_HERSHEY_PLAIN, 
                fontsize, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
        self.show_new_record -= 1

        cv2.imshow('LakituAI', image)
        cv2.waitKey(1)

if __name__ == '__main__':

    # a = LakituAI("D:\\Projects\\ClonkStobens\\mp4_download\\gcJH0qAQgnM.mp4",
    #             [0, 720, 306, 1280], # Game Coords
    #             [634, 684, 1173, 1223], # XCam Coords
    #             [225, 425, 610, 995], # Reset Coords
    #             [15, 75, 912, 1041], # Coin Coords
    #             [15, 75, 1118, 1240], # Star Coords
    #             "test",
    #             "2021-01-26",
    #             "ClintStevens",
    #             current_frame=265600)

    # a.watch()
    

    # SIMPLY 720P
    # frame = int((0 * 60 * 60 * 30) + (3 * 60 * 30) + (0 * 30))
    # RESET_TEMPLATE_1 = cv2.imread("templates\\simply_reset_1.jpg")
    # RESET_TEMPLATE_2 = cv2.imread("templates\\simply_reset_2.jpg")
    # a = LakituAI("D:\\Projects\\ClonkStobens\\simply_vods\\912777797.mp4",
    #             [34, 667, 396, 1257], # Game Coords
    #             [589, 636, 1165, 1211], # XCam Coords
    #             [235, 400, 670, 1003], # Reset Coords
    #             [45, 98, 928, 1045], # Coin Coords
    #             [45, 98, 1115, 1230], # Star Coords
    #             "912777797",
    #             "2021-02-13",
    #             "Simply",
    #             current_frame=frame,
    #             framerate=30)
    # a.watch()

    # SIMPLY 480P
    frame = int((0 * 60 * 60 * 30) + (47 * 60 * 30) + (18 * 30))
    # frame = int((4 * 60 * 60 * 30) + (44 * 60 * 30) + (9 * 30))
    RESET_TEMPLATE_1 = cv2.imread("templates\\simply_reset_1.jpg")
    RESET_TEMPLATE_2 = cv2.imread("templates\\simply_reset_2.jpg")
    a = LakituAI("D:\\Projects\\ClonkStobens\\simply_vods\\955173367.mp4",
                [22, 447, 264, 840], # Game Coords
                [394, 424, 775, 807], # XCam Coords
                [154, 269, 444, 669], # Reset Coords
                [30, 65, 623, 694], # Coin Coords
                [30, 65, 742, 819], # Star Coords
                "955173367",
                "2021-03-19",
                "Simply",
                "120 Star",
                "Twitch",
                current_frame=frame,
                framerate=30)
    a.watch()

    # PUNCAY 480P
    # frame = int((0 * 60 * 60 * 30) + (0 * 60 * 30) + (0 * 30))
    # RESET_TEMPLATE_1 = cv2.imread("templates\\puncay_reset_1.jpg")
    # RESET_TEMPLATE_2 = cv2.imread("templates\\puncay_reset_2.jpg")
    # a = LakituAI("D:\\Projects\\ClonkStobens\\simply_vods\\951028428.mp4",
    #             [0, 480, 199, 852], # Game Coords
    #             [420, 454, 781, 815], # XCam Coords
    #             [152,278, 407,660], # Reset Coords
    #             [12, 51, 607, 693], # Coin Coords
    #             [12, 51, 743, 828], # Star Coords
    #             "951028428",
    #             "2021-03-15",
    #             "Puncayshun",
    #             "120 Star",
    #             "Twitch",
    #             current_frame=frame,
    #             framerate=30)
    # a.watch()
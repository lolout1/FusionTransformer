from typing import List, Dict
import os
import copy
import numpy as np
import pandas as pd
from utils.loader  import DatasetBuilder
from sklearn.preprocessing import StandardScaler

SAMPLING_RATE = 200  # Hz (samples per second)
TARGET_DURATION = 12  # Target seconds for all instances
TARGET_SAMPLES = TARGET_DURATION * SAMPLING_RATE  # 12s * 200Hz = 2400 samples
TOLERANCE = 50 
TEST_YOUNG = ["SA03", "SA10" , "SA15", "SA20"]
TEST_ELDERLY = ["SE02" , "SE06", "SE10", "SE14"]
TEST_SUBJECTS = TEST_YOUNG + TEST_ELDERLY

class ModalityFile: 
    '''
    Represents an individual file in a modality, containing the subject ID, action ID, sequence number, and file path

    Attributes: 
    subject_id (int) : ID of the subject performing the action
    action_id (int) : ID of the action being performed
    sequence_number
    '''

    def __init__(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None: 
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

    def __repr__(self) -> str : 
        return (
            f"ModalityFile(subject_id  = {self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number}, file_path = '{self.file_path}')"
        )


class Modality:
    '''
    Represents a modality (e.g., RGB, Depth) containing a list of ModalityFile objects.

    Attributes:
        name (str): Name of the modality.
        files (List[ModalityFile]): List of files belonging to this modality.
    '''

    def __init__(self, name : str) -> None:
        self.name = name 
        self.files : List[ModalityFile] = []
    
    def add_file(self, subject_id: int , action_id: int, sequence_number: int, file_path: str) -> None: 
        '''
        Adds a file to the modality

        Args: 
            ubject_id (int): ID of the subject.
            action_id (int): ID of the action.
            sequence_number (int): Sequence number of the trial.
            file_path (str): Path to the file.
        '''
        modality_file = ModalityFile(subject_id, action_id, sequence_number, file_path)
        self.files.append(modality_file)
    
    def __repr__(self) -> str:
        return f"Modality(name='{self.name}', files={self.files})"
    

class MatchedTrial: 
    """
    Represents a matched trial containing files from different modalities for the same trial.

    Attributes:
        subject_id (int): ID of the subject.
        action_id (int): ID of the action.
        sequence_number (int): Sequence number of the trial.
        files (Dict[str, str]): Dictionary mapping modality names to file paths.
    """
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id  = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, List[str, ]] = {}
    
    def add_file(self, modality_name: str, file_path: str) -> None:
        '''
        Adds a file to the matched trial for a specific modality

        Args:
            modality_name (str) : Name of the modality
            file_path(str) : Path to the file
        '''
        self.files[modality_name] = file_path
    
    def __repr__(self) -> str:
        return f"MatchedTrial(subject_id={self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number}, files={self.files})"



class UTD_MHAD : 
    """
    Represents the UTD-MHAD dataset, managing the loading of files and matching of trials across modalities.

    Attributes:
        root_dir (str): Root directory of the UTD-MHAD dataset.
        modalities (Dict[str, Modality]): Dictionary of modality names to Modality objects.
        matched_trials (List[MatchedTrial]): List of matched trials containing files from different modalities.
    """

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.modalities: Dict[str, Modality] = {}
        self.matched_trials: List[MatchedTrial] = []
    
    def add_modality(self, modality_name: str) -> None:
        """
        Adds a modality to the dataset.

        Args:
            modality_name (str): Name of the modality.
        """
        self.modalities[modality_name] = Modality(modality_name)
    
    def load_files(self) -> None : 
        """
        Loads all files from the directory structure into their respective modalities.
        """
        for modality_name, modality in self.modalities.items():
            modality_dir = os.path.join(self.root_dir, modality_name)
            for root, _, files in os.walk(modality_dir):
                for file in files:
                    if file.endswith(('.avi', '.mp4', '.txt' , '.mat')):
                        # parts = root.split(os.sep)
                        subject_id = int(file.split('_')[1][1:])
                        action_id = int(file.split('_')[0][1:])
                        sequence_number = int(file.split('_')[2][1:])
                        file_path = os.path.join(root, file)
                        modality.add_file(subject_id, action_id, sequence_number, file_path)
    
    def match_trials(self) -> None: 
        '''
        Matches files from different modalities based on subject ID, action ID, and sequence number.
        '''

        for modality_name, modality in self.modalities.items():
            for modality_file in modality.files:
                matched_trial = self._find_or_create_matched_trial(modality_file.subject_id,
                                                                   modality_file.action_id,
                                                                   modality_file.sequence_number)
                
                matched_trial.add_file(modality_name, modality_file.file_path)

    def _find_or_create_matched_trial(self, subject_id: int, action_id: int, sequence_number: int) -> MatchedTrial:
        '''
        Finds or creates a MatchedTrial for a given subject ID, action ID, and sequence number.

        Args:
            subject_id (int): ID of the subject.
            action_id (int): ID of the action.
            sequence_number (int): Sequence number of the trial.

        Returns:
            MatchedTrial: The matched trial object.
        '''
        for trial in self.matched_trials:
            if (trial.subject_id == subject_id and trial.action_id==action_id \
                and trial.sequence_number == sequence_number):
                return trial
        new_trial = MatchedTrial(subject_id, action_id, sequence_number)
        self.matched_trials.append(new_trial)
        return new_trial



class SmartFallMM:
    """
    Represents the SmartFallMM dataset, managing the loading of files and matching of trials across modalities and specific sensors.

    Attributes:
        root_dir (str): Root directory of the SmartFallMM dataset.
        age_groups (Dict[str, Dict[str, Modality]]): Dictionary containing 'old' and 'young' groups, each having a dictionary of modality names to Modality objects.
        matched_trials (List[MatchedTrial]): List of matched trials containing files from different modalities.
        selected_sensors (Dict[str, str]): Dictionary storing selected sensors for modalities like 'accelerometer' and 'gyroscope'. Skeleton data is loaded as is.
    """

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.age_groups: Dict[str, Dict[str, Modality]] = {
            "old": {},
            "young": {}
        }
        self.matched_trials: List[MatchedTrial] = []
        self.selected_sensors: Dict[str, str] = {}  # Stores the selected sensor for each modality (e.g., accelerometer)

    def add_modality(self, age_group: str, modality_name: str) -> None:
        """
        Adds a modality to the dataset for a specific age group.

        Args:
            age_group (str): Either 'old' or 'young'.
            modality_name (str): Name of the modality (e.g., accelerometer, gyroscope, skeleton).
        """
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}. Expected 'old' or 'young'.")
        
        self.age_groups[age_group][modality_name] = Modality(modality_name)

    def select_sensor(self, modality_name: str, sensor_name: str = None) -> None:
        """
        Selects a specific sensor for a given modality if applicable. Only files from this sensor will be loaded for modalities like 'accelerometer' or 'gyroscope'.
        For modalities like 'skeleton', no sensor is needed.

        Args:
            modality_name (str): Name of the modality (e.g., accelerometer, gyroscope, skeleton).
            sensor_name (str): Name of the sensor (e.g., phone, watch, meta_wrist, meta_hip). None for 'skeleton'.
        """
        if modality_name == "skeleton":
            # Skeleton modality doesn't have sensor-specific data
            self.selected_sensors[modality_name] = None
        else:
            if sensor_name is None:
                raise ValueError(f"Sensor must be specified for modality '{modality_name}'")
            self.selected_sensors[modality_name] = sensor_name

    def load_files(self) -> None:
        """
        Loads files from the dataset based on selected sensors and age groups.
        Skeleton data is loaded without sensor selection.
        """
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                # Handle skeleton data (no sensor required)
                if modality_name == "skeleton":
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                else:
                    # Only load data from the selected sensor if it exists
                    if modality_name in self.selected_sensors:
                        sensor_name = self.selected_sensors[modality_name]
                        modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)
                    else:
                        continue

                # Load the files
                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        try :
                            if file.endswith(('.csv')):
                            # Extract information based on the filename
                                subject_id = int(file[1:3])  # Assuming S001 format for subject
                                action_id = int(file[4:6])  # Assuming A001 format for action
                                sequence_number = int(file[7:9])  # Assuming T001 format for trial
                                file_path = os.path.join(root, file)
                                modality.add_file(subject_id, action_id, sequence_number, file_path)
                        except:
                            print(file)


    def match_trials(self) -> None:
        """
        Matches files from different modalities based on subject ID, action ID, and sequence number.
        Only trials that have matching files in all modalities will be kept in matched_trials.
        """
        trial_dict = {}

        # Step 1: Group files by (subject_id, action_id, sequence_number)
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for modality_file in modality.files:
                    key = (modality_file.subject_id, modality_file.action_id, modality_file.sequence_number)

                    if key not in trial_dict:
                        trial_dict[key] = {}

                    # Add the file under its modality name
                    trial_dict[key][modality_name] = modality_file.file_path

        # Step 2: Filter out incomplete trials
        required_modalities = list(self.age_groups['young'].keys())  # Assuming all age groups have the same modalities

        for key, files_dict in trial_dict.items():
            # Check if all required modalities are present for this trial
            if all(modality in files_dict for modality in required_modalities):
                subject_id, action_id, sequence_number = key
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number)

                for modality_name, file_path in files_dict.items():
                    matched_trial.add_file(modality_name, file_path)

                self.matched_trials.append(matched_trial)


    def _find_or_create_matched_trial(self, subject_id: int, action_id: int, sequence_number: int) -> MatchedTrial:
        """
        Finds or creates a MatchedTrial for a given subject ID, action ID, and sequence number.

        Args:
            subject_id (int): ID of the subject.
            action_id (int): ID of the action.
            sequence_number (int): Sequence number of the trial.

        Returns:
            MatchedTrial: The matched trial object.
        """
        for trial in self.matched_trials:
            if (trial.subject_id == subject_id and trial.action_id == action_id
                    and trial.sequence_number == sequence_number):
                return trial
        new_trial = MatchedTrial(subject_id, action_id, sequence_number)
        self.matched_trials.append(new_trial)
        return new_trial

    def pipe_line(self, age_group : List[str], modalities: List[str], sensors: List[str]):
        '''
        A pipe to load the data 
        '''
        for age in age_group : 
                for modality in modalities:
                    self.add_modality(age, modality)
                    if modality == 'skeleton':
                        self.select_sensor('skeleton')
                    else: 
                        for sensor in sensors:
                            self.select_sensor(modality, sensor)

        # Load files for the selected sensors and skeleton data)
        self.load_files()

        # Match trials across the modalities
        self.match_trials()



def prepare_smartfallmm(arg )  -> DatasetBuilder:
    '''
    Function for dataset preparation
    '''
    dataset_args = copy.deepcopy(arg.dataset_args)
    use_skeleton = dataset_args.get('use_skeleton', True)
    modalities = dataset_args.get('modalities', [])
    if not use_skeleton:
        modalities = [modality for modality in modalities if modality != 'skeleton']
    dataset_args['modalities'] = modalities

    sm_dataset = SmartFallMM(root_dir=os.path.join(os.getcwd(), 'data'))
    sm_dataset.pipe_line(age_group=dataset_args['age_group'], \
                        modalities=modalities, \
                        sensors=dataset_args['sensors'])
    # Pass all dataset_args to DatasetBuilder to support filtering and other options
    builder = DatasetBuilder(sm_dataset, **dataset_args)
    return builder

def split_by_subjects(builder, subjects, fuse) -> Dict[str, np.ndarray]:
    '''
    Function to Filter out expects subjects
    '''
    builder.make_dataset(subjects, fuse)

    # if len(subjects) == 1 and subjects[0] in [35, 38,39,44, 46]: 
    #     builder.random_resampling()
    norm_data = builder.normalization()
    #norm_data = builder.data
    return norm_data

def extract_subject_id(file_name):
    return file_name.split("_")[1]

def get_label(file_name): 
    return 1 if file_name.startswith("F") else 0


def load_sisfall(test_subjects=TEST_SUBJECTS):
    X_train, X_test, y_train, y_test = [], [], [], []
    data_dir = f"{os.getcwd()}/data/SisFall_dataset_processed"
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"): 
            file_path = os.path.join(data_dir, file_name)

            df = pd.read_csv(file_path, sep = ',', header = None)
            if df.shape[0] != 2400 :
                continue
            label = get_label(file_name)
            subject_id = extract_subject_id(file_name)
            if subject_id in test_subjects: 
                X_test.append(df.values[:, :3])
                y_test.append(label)
            else: 
                X_train.append(df.values[:, :3])
                y_train.append(label)

    X_train, X_test = np.array(X_train).astype(np.float64), np.array(X_test).astype(np.float64)
    y_train, y_test = np.array(y_train).astype(np.float64), np.array(y_test).astype(np.float64)

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
#     dataset = SmartFallMM(root_dir=os.path.join(os.getcwd(), 'data/smartfallmm'))

# # Add modalities for 'young' age group
#     dataset.add_modality("young", "accelerometer")
#     dataset.add_modality("young", "skeleton")

#     # Add modalities for 'old' age group
#     dataset.add_modality("old", "accelerometer")
#     dataset.add_modality("old", "skeleton")

#     # Select the sensor type for accelerometer and gyroscope
#     dataset.select_sensor("accelerometer", "phone")

#     # For skeleton, no sensor needs to be selected
#     dataset.select_sensor("skeleton")

#     # Load files for the selected sensors and skeleton data
#     dataset.load_files()

#     # Match trials across the modalities
#     dataset.match_trials()
#     # take input 
#     # load files 
#         # load files only that matches 
#     # merge all together
#     # create labels
    (X_train, y_train), (X_test, y_test) = load_sisfall()
    print(X_train.shape)

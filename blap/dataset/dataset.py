import pandas as pd
from torch.utils.data import Dataset
from random import randint
import numpy as np

class ShutterStock(Dataset):
    """
    ShutterStock Dataset
    
    A custom dataset to handle music data stored in an Excel file. For each entry, 
    it selects a random music chunk and its corresponding caption.
    
    Attributes:
    - dataframe (pd.DataFrame): The dataframe containing the music data.
    - musicPath (str): Path to the directory containing music chunks.
    """
    
    def _convert_to_seconds(self, time_str):
        """
        Converts a time string in the format 'MM:SS:ms' to seconds.
        
        Args:
        - time_str (str): The time string in the format 'MM:SS:ms'.
        
        Returns:
        - float: The time in seconds.
        """
        minutes, seconds, milliseconds = map(int, time_str.split(':'))
        total_seconds = minutes * 60 + seconds + milliseconds / 1000
        return total_seconds

    def _numberChunks(self, length):
        """
        Calculate the number of 15-second chunks for a given length.
        
        Args:
        - length (float): The length in seconds.
        
        Returns:
        - int: The number of chunks.
        """
        return int(length / 15)

    def _chunkName(self, fileName, ID):
        """
        Generate a chunk filename based on the original filename and chunk ID.
        
        Args:
        - fileName (str): The original filename without extension.
        - ID (int): The chunk ID.
        
        Returns:
        - str: The chunk filename.
        """
        return fileName + f"/chunk{ID}.npy"

    def _selectRandomChunkCaptionCombo(self, time_str):
        """
        Select a random chunk and caption combination based on the duration.
        
        Args:
        - time_str (str): The duration in the format 'MM:SS:ms'.
        
        Returns:
        - tuple: (chunkID, captionID) where chunkID is the chunk index 
                 and captionID is the index of the caption.
        """
        chunks = self._numberChunks(self._convert_to_seconds(time_str))
        chunk = randint(1, chunks)
        caption = randint(0, 2)
        return (chunk, caption)

    def __init__(self, dataFile, musicPath) -> None:
        """
        Initialize the ShutterStock dataset.
        
        Args:
        - dataFile (str): Path to the Excel file containing music data.
        - musicPath (str): Path to the directory containing music chunks.
        """
        super().__init__()
        self.dataframe = pd.read_excel(dataFile)
        self.musicPath = musicPath

    def __len__(self):
        """
        Get the number of entries in the dataset.
        
        Returns:
        - int: The number of entries.
        """
        return len(self.dataframe)
    
    def __getitem__(self, index):
        """
        Fetch a random music chunk and its corresponding caption for a given index.
        
        Args:
        - index (int): The index of the data entry.
        
        Returns:
        - tuple: (chunkName, caption) where chunkName is the path to the music chunk 
                 and caption is the selected caption text.
        """
        entry = self.dataframe.iloc[index]
        (chunkID, captionID) = self._selectRandomChunkCaptionCombo(entry["Duration"])

        chunkName = self._chunkName(str(entry["ID"]), chunkID)
        chunkName = self.musicPath + "/" + chunkName
        if captionID == 0:
            caption = entry["Description"]
        elif captionID == 1:
            caption = entry["OpenAI"]
        else:
            caption = entry["ETH GPT"]

        return (np.load(chunkName, allow_pickle=True), caption)


class MusicCaps(Dataset):
    """
    MusicCaps Dataset
    
    A custom dataset to handle music data stored in an Excel file. For each entry, 
    it loads an audio file in the form of a NumPy array and its corresponding caption.
    
    Attributes:
    - dataframe (pd.DataFrame): The dataframe containing the music data.
    - musicPath (str): Path to the directory containing audio files in the .npy format.
    """

    def __init__(self,dataFile=None, musicPath=None, dataframe=None) -> None:
        """
        Initialize the MusicCaps dataset.
        
        Args:
        - dataFile (str, optional): Path to the Excel file containing music data.
        - musicPath (str, optional): Path to the directory containing audio files in the .npy format.
        - dataframe (pd.DataFrame, optional): A DataFrame containing the music data.
        """
        super().__init__()

        if dataframe is not None:
            self.dataframe = dataframe
        else:
            self.dataframe = pd.read_excel(dataFile)

        self.musicPath = musicPath
    
    @classmethod
    def from_dataframe(cls, dataframe, musicPath):
        """
        Alternative constructor to initialize the MusicCaps dataset directly from a DataFrame.

        Args:
        - dataframe (pd.DataFrame): A DataFrame containing the music data.
        - musicPath (str): Path to the directory containing audio files in the .npy format.

        Returns:
        - MusicCaps: An instance of the MusicCaps class.
        """

        return cls(dataframe=dataframe, musicPath=musicPath)

    def __len__(self):
        """
        Get the number of entries in the dataset.
        
        Returns:
        - int: The number of entries.
        """
        return len(self.dataframe)
    
    def __getitem__(self, index):
        """
        Fetch an audio file (as a NumPy array) and its corresponding caption for a given index.
        
        Args:
        - index (int): The index of the data entry.
        
        Returns:
        - tuple: (audio_data, caption) where audio_data is the loaded NumPy array of the audio file 
                 and caption is the corresponding caption text.
        """
        entry = self.dataframe.iloc[index]
        audioNPY = entry["ID"]
        caption  = entry["Description"]

        audioNPY = self.musicPath + "/" + audioNPY + ".npy"
        return np.load(audioNPY, allow_pickle=True), caption
    
    def createSplit(self):
        """
        Create a split of the MusicCaps dataset into training and validation subsets.

        This method divides the dataset based on a boolean column 'is_audioset_eval' in the DataFrame. 
        Rows with `True` in this column are treated as part of the validation set, and the rest as 
        part of the training set.

        Returns:
        - tuple: A tuple containing two `MusicCaps` instances:
            - train (MusicCaps): The training subset of the dataset.
            - val (MusicCaps): The validation subset of the dataset.
        """
        df_val = self.dataframe[self.dataframe["is_audioset_eval"]==1.0]
        df_train = self.dataframe[self.dataframe["is_audioset_eval"]==0.0]

        train = MusicCaps.from_dataframe(df_train, self.musicPath)
        val = MusicCaps.from_dataframe(df_val, self.musicPath)
        return train, val

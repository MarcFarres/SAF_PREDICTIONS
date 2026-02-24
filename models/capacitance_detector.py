import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CapacitanceDetector():
    """
    A soil capacitance detector
    """
    
    REQUIRED_COLUMNS = ["date", "soil_moisture_40"]
    
    def __init__(self, beta : float = 0.3):
        """
        Initialize the detector with a given beta
        
        Parameters
        ---------------
        beta : float
            Parameter for moving average. It gives the weight of the previous result.
        """
        self.BETA = beta
        
    def _check_dataframe(self, df : pd.DataFrame):
        
        if not all(col in df.columns for col in self.REQUIRED_COLUMNS):
            raise ValueError(f"Missing required columns: {self.REQUIRED_COLUMNS}")
        
    def _process_dataset(self, df : pd.DataFrame) -> pd.DataFrame:
        
        df_processed = df
        df_processed["difference"] = np.concatenate([[np.nan], np.diff(df_processed["soil_moisture_40"])])
        df_processed['difference2'] = np.concatenate([[np.nan], np.diff(df_processed['difference'])])
        
        return df_processed
        
    def detect_capacitances(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Detects the capacitances of the whole dataframe and returns their values and dates
        
        Parameters
        -----------
        df : pd.Dataframe
            A dataframe that must contain soil_moisture_40 and its associated measure timestaps (`[soil_moisture_40, date]`).
            
        Returns
        -----------
        capacitancy_frame : pd.Dataframe
            A dataframe listing capacitance values extracted from the data, along with the dates on which each value was found.
        """
        
        self._check_dataframe(df)
        df_processed = self._process_dataset(df)
        
        gradient_marker = []
        up_found = False
        down_found = False

        capacitancy_points = []

        found = 0
        
        logger.info("Beginning capacitance detection")
        
        #We check if we are on our first descend after irrigation, and if so
        #we look when the second derivative (the curvature of the moisture)
        #changes from negative to positive (from descend to plateau).
        for row in df_processed.iloc[1:].itertuples(index=True):
            
            down = row.irrigation_volume_0 == 0
            
            if not up_found and not down:
                up_found = True
                continue
            
            if up_found and not down_found and down:
                down_found = True
                value = row.difference2
                found = 0
                continue
            
            if up_found and down_found:
                prev_value = value
                value = row.difference2
                
                #We check that the derivative is mantained to avoid noisy lectures
                if (prev_value < 0 and value >= 0) and found == 0:
                    found += 1
                elif found > 0 and value >= 0:
                    found += 1
                else:
                    found = 0
                    
                if found == 3:
                    gradient_marker.append(row.date)
                    up_found = False
                    down_found = False
                    #We do a moving average with the capacitancy points, since
                    #they should change slowly in time. We ensure some degree
                    #of smoothness this way.
                    if len(capacitancy_points) > 0:
                        capacitancy_points.append((row.date, self.BETA*capacitancy_points[-1][1] + (1 - self.BETA)*row.soil_moisture_40))
                    else:
                        capacitancy_points.append((row.date, row.soil_moisture_40))
                    
        capacitancy_frame = pd.DataFrame(capacitancy_points, columns=["date", "capacitancy"])
        
        logger.info("Capacitance detection complete")
        
        return capacitancy_frame
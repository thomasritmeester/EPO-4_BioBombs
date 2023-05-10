

const int ECG=A0;
const int EMG=A2;
const int GSR=A1;
int ecgValue=0;
int emgValue=0;
int gsrValue=0;
long ecg_sum=0;
long emg_sum=0;
long gsr_sum=0;
int gsr_average=0;
int ecg_average=0;
int emg_average=0;
int emg_cal_peak;
int ecg_cal_peak;
int emg_cal_peak_sum;
int ecg_cal_peak_sum;
float ratio;

float calibrate(){
  emg_sum=0;
  ecg_sum=0;
  for(int k=0; k<4;k++)
    {
      emg_cal_peak=0;
      ecg_cal_peak=0;
      for(int j=0;j<1000;j++)
        {
            emgValue=analogRead(EMG);
            ecgValue=analogRead(ECG);
            delay(5);
          if (emg_cal_peak < emgValue){
            emg_cal_peak = emgValue;
          }
          if (ecg_cal_peak < ecgValue){
            ecg_cal_peak = ecgValue;
          }
          emg_sum += emgValue;
          ecg_sum += ecgValue;
        }
        
      emg_cal_peak_sum += emg_cal_peak;
      ecg_cal_peak_sum += ecg_cal_peak;
    }

    ratio = (float)(emg_cal_peak_sum/4-(emg_sum/4000))/(float)(ecg_cal_peak_sum/4-(ecg_sum/4000));
    Serial.print("Ratio:");
    Serial.println(ratio);
  return (ratio);
}



void setup(){
  Serial.begin(9600);
  calibrate();

}

void loop(){
  long ecg_sum=0, gsr_sum=0, HR=0;
  for(int i=0;i<10;i++)           //Average the 10 measurements to remove the glitch
      {
      gsrValue=analogRead(EMG);
      gsr_sum += gsrValue;
      ecgValue=analogRead(ECG);
      ecg_sum += ecgValue;
      delayMicroseconds(500);
      }
  gsr_average = gsr_sum/10;
  ecg_average = ecg_sum/10;

  String SerialString = String(128);  
  Serial.print("GSR average:");
  Serial.print(gsr_average);
  Serial.print(". ECG average: ");
  Serial.print(ecg_average);
  Serial.println(".");
}

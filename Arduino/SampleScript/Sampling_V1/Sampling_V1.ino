const int ECG=A0;
const int GSR=A1;
int ecgValue=0;
int gsrValue=0;
int gsr_average=0;
int ecg_average=0;

void setup(){
  Serial.begin(9600);
}

void loop(){
  long ecg_sum=0, gsr_sum=0, HR=0;
  for(int i=0;i<10;i++)           //Average the 10 measurements to remove the glitch
      {
      gsrValue=analogRead(GSR);
      gsr_sum += gsrValue;
      ecgValue=analogRead(ECG);
      ecg_sum += ecgValue;
      delay(5);
      }
  gsr_average = gsr_sum/10;
  ecg_average = ecg_sum/10;
  HR = ((1024+2*gsr_average)*10000)/(512-gsr_average);

  String SerialString = String(128);  
  Serial.print("GSR average:");
  Serial.print(gsr_average);
  Serial.print(". Human Resistance: ");
  Serial.print(HR);
  Serial.print(". ECG average: ");
  Serial.print(ecg_average);
  Serial.println(".");
}
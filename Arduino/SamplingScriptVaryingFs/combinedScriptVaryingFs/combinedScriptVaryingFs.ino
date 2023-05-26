//Script sampling the different values with different sample rates

#include "SerialTransfer.h"
#include "I2Cdev.h"
#include "Wire.h"
#include <Adafruit_MLX90614.h>

const int MPU_addr[] = {0x68, 0x69};
Adafruit_MLX90614 tempSensor = Adafruit_MLX90614();
SerialTransfer myTransfer;

const int ECG = A0;
const int GSR = A1;
const int EMG = A2;
//ACC sensor is connected to A4 and A5 using the I2C protocol

uint16_t ecg_value = 0;
uint16_t gsr_value = 0;
uint16_t emg_value = 0;

int16_t ax[] = {0,0};
int16_t ay[] = {0,0};
int16_t az[] = {0,0};

uint16_t temperature;

struct STRUCT {
  unsigned long timestamp;
  uint16_t  ECG_data;
  uint16_t GSR_data;
  uint16_t EMG_data;
  int16_t ACC1_X_data;
  int16_t ACC1_Y_data;
  int16_t ACC1_Z_data;
  int16_t ACC2_X_data;
  int16_t ACC2_Y_data;
  int16_t ACC2_Z_data;
  uint16_t TEMP_DATA;
} dataPacket;

void setup() {
  //starts serial connection
  Serial.begin(57600);
  myTransfer.begin(Serial);
  //initializes the temperature sensor
  tempSensor.begin();
//setup I2C transmission for accelerometer
  Wire.begin();
  for(int i=0; i<2; i++){
    Wire.beginTransmission(MPU_addr[i]);
    Wire.write(0x6B);  // PWR_MGMT_1 register
    Wire.write(0);     // set to zero (wakes up the MPU-6050)
    Wire.endTransmission(true);
  }
  Serial.begin(9600);

//Hardware interrupt settings for constant sample rate
  cli();

//set timer0 interrupt at 710Hz. Used for ECG, EMG, GSR
  TCCR0A = 0;// set entire TCCR0A register to 0
  TCCR0B = 0;// same for TCCR0B
  TCNT0  = 0;//initialize counter value to 0
  // set compare match register for 2khz increments
  OCR0A = 22;// = (16*10^6) / (710*1024) - 1 (must be <256)
  // turn on CTC mode
  TCCR0A |= (1 << WGM01);
  // Set CS02 and CS00 bits for 1024 prescaler
  TCCR0B |= (1 << CS02) | (1 << CS00);   
  // enable timer compare interrupt
  TIMSK0 |= (1 << OCIE0A);

//set timer1 interrupt at 100Hz, used for ACC, Temp
  TCCR1A = 0;// set entire TCCR1A register to 0
  TCCR1B = 0;// same for TCCR1B
  TCNT1  = 0;//initialize counter value to 0
  // set compare match register for 100hz increments
  OCR1A = 156;// = (16*10^6) / (100*1024) - 1 (must be <65536)
  // turn on CTC mode
  TCCR1B |= (1 << WGM12);
  // Set CS10 and CS12 bits for 1024 prescaler
  TCCR1B |= (1 << CS12) | (1 << CS10);  
  // enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);

  sei();//allow interrupts

}//end setup


void loop() {
}

ISR(TIMER0_COMPA_vect){
  takeMeasurementsHighFs();
  sendData();
}

ISR(TIMER1_COMPA_vect){
  takeMeasurementsLowFs();
}

void takeMeasurementsHighFs() {
  gsr_value = analogRead(GSR);
  ecg_value = analogRead(ECG);
  emg_value = analogRead(EMG);
}

void takeMeasurementsLowFs() {
  //measure accelerometer values
  for(int i=0; i<2; i++){
      Wire.beginTransmission(MPU_addr[i]);
      Wire.write(0x3B);  // starting with register 0x3B (ACCEL_XOUT_H)
      Wire.endTransmission(false);
      Wire.requestFrom(MPU_addr, 6, true); // request a total of 6 registers
      int t = Wire.read();
      ax[i] = (t << 8) | Wire.read(); // 0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)
      t = Wire.read();
      ay[i] = (t << 8) | Wire.read(); // 0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
      t = Wire.read();
      az[i] = (t << 8) | Wire.read(); // 0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)
      //t = Wire.read();
  }

  temperature = tempSensor.readObjectTempC()*100;
}

void sendData() {
  uint16_t sendSize = 0;

  dataPacket.timestamp = millis();
  dataPacket.ECG_data = gsr_value;
  dataPacket.GSR_data = ecg_value;
  dataPacket.EMG_data = emg_value;
  dataPacket.ACC1_X_data = ax[0];
  dataPacket.ACC1_Y_data = ay[0];
  dataPacket.ACC1_Z_data = az[0];
  dataPacket.ACC2_X_data = ax[1];
  dataPacket.ACC2_Y_data = ay[1];
  dataPacket.ACC2_Z_data = az[1];
  dataPacket.TEMP_DATA = temperature;

  sendSize = myTransfer.txObj(dataPacket, sendSize);
  // Send buffer
  myTransfer.sendData(sendSize);
}

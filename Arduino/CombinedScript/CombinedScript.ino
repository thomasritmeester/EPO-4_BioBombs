#include "SerialTransfer.h"
#include "I2Cdev.h"
#include "MPU6050.h"

// Arduino Wire library is required if I2Cdev I2CDEV_ARDUINO_WIRE implementation
// is used in I2Cdev.h
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
#include "Wire.h"
#endif

MPU6050 mpu;
SerialTransfer myTransfer;

int Fs = 100; //Hz
unsigned long prevMillis;

const int ECG = A0;
const int GSR = A1;
const int EMG = A2;
//ACC sensor is connected to A4 and A5 using the I2C protocol

uint16_t ecgValue = 0;
uint16_t gsrValue = 0;
uint16_t emgValue = 0;

uint16_t gsr_average = 0;
uint16_t ecg_average = 0;
uint16_t emg_average = 0;

int16_t ax, ay, az, gx, gy, gz;

struct STRUCT {
  unsigned long timestamp;
  uint16_t  ECG_data;
  uint16_t GSR_data;
  int16_t EMG_data;
  int16_t ACC_X_data;
  int16_t ACC_Y_data;
  int16_t ACC_Z_data;
} dataPacket;

void setup()
{
  // join I2C bus (I2Cdev library doesn't do this automatically)
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
  Wire.begin();
#elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
  Fastwire::setup(400, true);
#endif

  Serial.begin(57600);
  myTransfer.begin(Serial);

  //initialize accelerometer
  mpu.initialize();
  mpu.setXGyroOffset(59);
  mpu.setYGyroOffset(-24);
  mpu.setZGyroOffset(1724);
  
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
}

void loop()
{
  takeMeasurements();

  // hold the program until 1/Fs has passed since the previous data packet was sent
  while (1) {
    if ((millis() - prevMillis) >= (1000 / Fs)) {
      prevMillis = millis();
      break;
    }
  }
  //send the actual data
  sendData();
}


void takeMeasurements() {
  //take 10 measurements and average them
  int ecg_sum = 0, gsr_sum = 0, emg_sum = 0;
  for (int i = 0; i < 10; i++)
  {
    gsrValue = analogRead(GSR);
    gsr_sum += gsrValue;
    ecgValue = analogRead(ECG);
    ecg_sum += ecgValue;
    emgValue = analogRead(EMG);
    emg_sum += emgValue;
    delayMicroseconds(500);
  }
  gsr_average = gsr_sum / 10;
  ecg_average = ecg_sum / 10;
  emg_average = emg_sum / 10;

  //measure accelerometer values
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
}

void sendData() {
  uint16_t sendSize = 0;

  dataPacket.timestamp = millis();
  dataPacket.ECG_data = gsr_average;
  dataPacket.GSR_data = ecg_average;
  dataPacket.EMG_data = emg_average;
  dataPacket.ACC_X_data = ax;
  dataPacket.ACC_Y_data = ay;
  dataPacket.ACC_Z_data = az;

  sendSize = myTransfer.txObj(dataPacket, sendSize);
  // Send buffer
  myTransfer.sendData(sendSize);
}

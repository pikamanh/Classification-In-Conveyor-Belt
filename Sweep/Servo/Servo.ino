/* Sweep
 by BARRAGAN <http://barraganstudio.com>
 This example code is in the public domain.

 modified 8 Nov 2013
 by Scott Fitzgerald
 https://www.arduino.cc/en/Tutorial/LibraryExamples/Sweep
*/

#include <Servo.h>

Servo myservo;  // create Servo object to control a servo
// twelve Servo objects can be created on most boards

int pos = 90;    // variable to store the servo position

void setup() {
  Serial.begin(9600);
  myservo.attach(9);
  myservo.write(90);
}

const long delayTime_1 = 3000;
const long delayTime_2 = 1000;

void loop() {
  if (Serial.available() > 0) {
        char command = Serial.read();  // Nhận tín hiệu từ máy tính
        if (command == '1') {       //Left
          unsigned long currentMillis = millis();
          while (millis() - currentMillis < delayTime_1)
          {

          }
          for (pos = 90; pos <= 180; pos += 10) {
          // in steps of 5 degree
            myservo.write(pos);              
            delay(20);                       
          }

          delay(1000);

          for (pos = 180; pos >= 90; pos -= 10) { 
            myservo.write(pos);              
            delay(20);                       
          }
          Serial.println("R done");
        }

        if (command == '2') {       //Right
          unsigned long currentMillis = millis();
          while (millis() - currentMillis < delayTime_2)
          {
            
          }
          for (pos = 90; pos >= 0; pos -= 10) {
          // in steps of 5 degree
            myservo.write(pos);              
            delay(20);                       
          }

          delay(1000);

          for (pos = 0; pos <= 90; pos += 10) {
            myservo.write(pos);              
            delay(20);                       
          }
          Serial.println("T done");
        }
    }
}

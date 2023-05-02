#include <Arduino.h>

// LEDS and BUTTONS
const int GREEN=4;
const int BLUE=5;
const int RED=6;
const int SW1=2;
const int SW2=3;

// LED states
bool green_state = false;
bool red_state = false;

// LED delays
int green_led_delay=500;
int red_led_delay=500;

// button variables
int button1_presses = 0;
int button2_presses = 0;
bool double_click_detected = false;

// Timer variables
hw_timer_t * timer = NULL;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

void IRAM_ATTR onTimer() {
  portENTER_CRITICAL_ISR(&timerMux);

  static int led_toggle = 0;
  static bool led_on = false;

  // If both buttons are pressed
  if (green_state && red_state) {
    led_toggle = 0;
    led_on = true;
    digitalWrite(GREEN, led_on);
    digitalWrite(RED, led_on);
  } 
  // If only green button is pressed
  else if (green_state && !red_state) {
    led_toggle = GREEN;
  } 
  // If only red button is pressed
  else if (!green_state && red_state) {
    led_toggle = RED;
  } 
  // If no buttons are pressed
  else {
    led_toggle = 0;
    led_on = false;
    digitalWrite(GREEN, led_on);
    digitalWrite(RED, led_on);
  }

  // Toggle the LED if necessary
  if (led_toggle != 0) {
    digitalWrite(led_toggle, led_on);
    led_on = !led_on;
  }

  portEXIT_CRITICAL_ISR(&timerMux);
}

void IRAM_ATTR onButton1Press() {
  button1_presses++;
  green_state = button1_presses % 2 == 0;
}

void IRAM_ATTR onButton2Press() {
  button2_presses++;
  red_state = button2_presses % 2 == 0;

  // Check if double click is detected
  if (!double_click_detected) {
    double_click_detected = true;
    digitalWrite(BLUE, HIGH);
    delay(500);
    digitalWrite(BLUE, LOW);
  } else {
    double_click_detected = false;
    green_led_delay = green_led_delay == 500 ? 250 : 500;
  }
}

void setup() {
  pinMode(GREEN, OUTPUT);
  pinMode(BLUE, OUTPUT);
  pinMode(RED, OUTPUT);
  pinMode(SW1, INPUT_PULLUP);
  pinMode(SW2, INPUT_PULLUP);

  // Interrupts for buttons
  attachInterrupt(digitalPinToInterrupt(SW1), onButton1Press, FALLING);
  attachInterrupt(digitalPinToInterrupt(SW2), onButton2Press, FALLING);

  // Initialize timer
  timer = timerBegin(0, 80, true);
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, green_led_delay, true);
  timerAlarmEnable(timer);
}

void loop() {
  // Put any code you want to run repeatedly here
}

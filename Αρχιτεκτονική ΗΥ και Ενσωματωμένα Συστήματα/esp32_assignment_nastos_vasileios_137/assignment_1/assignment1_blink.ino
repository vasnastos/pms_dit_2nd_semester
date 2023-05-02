// LEDS and BUTTONS
const int GREEN=4;
const int BLUE=5;
const int RED=6;
const int SW1=2;
const int SW2=3;


//  button and led variables
int buttonSwitch1 = 0;
int buttonSwitch2 = 0;
int button1_pressed = 0;
int button2_pressed = 0;
int green_state = 0;
int red_state = 0;
unsigned long firstClickTime;
bool firstClick = false;

// LED delays
int green_led_delay=500;
int red_led_delay=500;
int fdelay=-1;  // Detarmines the delay at each step on the turn_on_off function
int green_led_operation=0;  // if green led operation==1 then green_led_delay/2 else  green_led_delay*2


void setup() {
  pinMode(GREEN, OUTPUT); //Green Led
  pinMode(BLUE, OUTPUT); //Blue Led
  pinMode(RED, OUTPUT);  //Red Led
  pinMode(SW1, INPUT_PULLUP); //SW1 as input
  pinMode(SW2, INPUT_PULLUP); //SW2 as input
  Serial.begin(9600);
}

void button1_click()
{
  if (buttonSwitch1 == LOW) 
  { 
    delay(10);
    button1_pressed++;
  }  
}

void button2_click()
{
  if (buttonSwitch2 == LOW) {
    if (!firstClick) {
      firstClick = true;
      firstClickTime = millis();
    } else {
      // Detect single click
      if (millis() - firstClickTime < 500) {
        // Double click detected
        digitalWrite(BLUE, HIGH);  // turn the LED on (HIGH is the voltage level)
      } else {
        // Single click
        button2_pressed++;
        green_led_operation=!green_led_operation;
        green_led_delay=green_led_operation==1?green_led_delay/2:green_led_delay*2;
        digitalWrite(BLUE, LOW);  // turn the LED on (HIGH is the voltage level)
      }
      firstClick = false;
    }
  }
}


void turn_on_off(int LED)
{
  fdelay=LED==GREEN?green_led_delay:red_led_delay;
  digitalWrite(LED,HIGH);
  delay(fdelay);
  digitalWrite(LED,LOW);
  delay(fdelay);
}

void set_led_states()
{
  green_state=button1_pressed%2==0?1:0;
  red_state=button2_pressed%2==0?1:0;
}

void loop() {
  // put your main code here, to run repeatedly:
  buttonSwitch1 = digitalRead(SW1);
  buttonSwitch2 = digitalRead(SW2);

  button1_click();
  button2_click();
  set_led_states();


  if (green_state ==1 && red_state ==1)
  {
    digitalWrite(GREEN, HIGH);  // turn the LED on (HIGH is the voltage level)
    digitalWrite(RED, HIGH);  // turn the LED on (HIGH is the voltage level)
    delay(green_led_delay);                     
    digitalWrite(GREEN, LOW);  // turn the LED off 
	if(green_led_delay==250)
	{
		delay(green_led_delay);
	}
    digitalWrite(RED, LOW);   // turn the LED off by making the voltage LOW           
  }
  else if (green_state ==1 && red_state ==0) 
  {
    turn_on_off(GREEN);
  }
  else if (green_state ==0 && red_state ==1) 
  {
    turn_on_off(RED);
  }
  else
  {
    digitalWrite(GREEN, LOW);  // turn the LED on (HIGH is the voltage level)
    digitalWrite(RED, LOW);  // turn the LED on (HIGH is the voltage level)
  }
}

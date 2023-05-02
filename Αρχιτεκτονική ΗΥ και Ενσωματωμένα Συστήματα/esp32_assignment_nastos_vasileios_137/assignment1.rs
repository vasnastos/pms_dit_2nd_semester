use esp_idf_sys::{gpio_set_direction, gpio_set_level, gpio_get_level};
use esp_idf_sys::gpio::{GPIO_NUM_17, GPIO_NUM_18, GPIO_MODE_OUTPUT};
use esp_idf_sys::gpio_pad_select_gpio;
use esp_idf_sys::esp_timer::esp_timer_get_time;

// Define constants for the LED pins and button switches
const RED_LED: u8 = GPIO_NUM_17;
const GREEN_LED: u8 = GPIO_NUM_18;
const SW1: u8 = GPIO_NUM_19;
const SW2: u8 = GPIO_NUM_20;

// Set the initial frequency of dimming
let dim_frequency: u32 = 1000; // 1000ms


fn main() {
    // Configure the LEDs as output pins
    gpio_pad_select_gpio(RED_LED);
    gpio_set_direction(RED_LED, GPIO_MODE_OUTPUT);
    gpio_pad_select_gpio(GREEN_LED);
    gpio_set_direction(GREEN_LED, GPIO_MODE_OUTPUT);

    // Configure the button switches as input pins
    gpio_pad_select_gpio(SW1);
    gpio_set_direction(SW1, GPIO_MODE_INPUT);
    gpio_pad_select_gpio(SW2);
    gpio_set_direction(SW2, GPIO_MODE_INPUT);

    // Initialize the LEDs to be off
    gpio_set_level(RED_LED, 0);
    gpio_set_level(GREEN_LED, 0);

    loop {
        // Check the state of SW1
        if gpio_get_level(SW1) == 0 {
            // SW1 is pressed, turn off the red LED
            gpio_set_level(RED_LED, 0);
        } else {
            // SW1 is not pressed, dim the red LED
            gpio_set_level(RED_LED, 1);
            delay_ms(dim_frequency);
            gpio_set_level(RED_LED, 0);
            delay_ms(dim_frequency);
        }

        // Check the state of SW2
        if gpio_get_level(SW2) == 0 {
            // SW2 is pressed, dim the green LED
            gpio_set_level(GREEN_LED, 1);
            delay_ms(dim_frequency);
            gpio_set_level(GREEN_LED, 0);
            delay_ms(dim_frequency);
        } else {
            // SW2 is not pressed, check if it has been double clicked
            let current_time = esp_timer_get_time();
            while gpio_get_level(SW2) != 0 {
                // wait for the button to be released
            }
            let release_time = esp_timer_get_time();
            if release_time - current_time <= 2000000 {
                // button was double clicked, double the dimming frequency
                dim_frequency /= 2;
            }
        }
    }
}
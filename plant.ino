#include <DHT.h>
#include <SoftwareSerial.h>

// =====================================================
//                 PIN CONFIGURATION
// =====================================================

#define SOIL_PIN A0
#define LDR_PIN  A1

#define DHT_PIN 2
#define DHT_TYPE DHT11

#define RELAY_PIN 7

// GSM
#define GSM_RX 10     // Arduino RX <- GSM TX
#define GSM_TX 11     // Arduino TX -> GSM RX


// =====================================================
//                    OBJECTS
// =====================================================

DHT dht(DHT_PIN, DHT_TYPE);
SoftwareSerial gsm(GSM_RX, GSM_TX);


// =====================================================
//                  PHONE NUMBER
// =====================================================

const char phoneNumber[] = "+918918952076";


// =====================================================
//              MOISTURE THRESHOLDS
// =====================================================

// Pump starts below this moisture
const int WATER_START = 35;

// Pump stops at this moisture
const int WATER_STOP = 60;


// =====================================================
//             SOIL MOISTURE CALIBRATION
// =====================================================

// Dry sensor = 1023
// Wet sensor = 0

const int SOIL_DRY_RAW = 1023;
const int SOIL_WET_RAW = 0;


// =====================================================
//                 TIMING SETTINGS
// =====================================================

// Sensor data sent to Python every 1 second
const unsigned long SENSOR_INTERVAL = 1000;

// Maximum time pump is allowed to run
// 30 minutes = 1800000 ms
const unsigned long MAX_PUMP_RUNTIME = 1800000UL;


// =====================================================
//                    VARIABLES
// =====================================================

bool pumpState = false;

// Moisture before watering
int previousMoisture = 0;

// Current moisture
int currentMoisture = 0;

// Latest humidity
float humidity = 0;

// Latest light level
int lightLevel = 0;

// Raw sensor values
int soilRaw = 0;
int lightRaw = 0;

// Timing
unsigned long lastSensorRead = 0;
unsigned long pumpStartTime = 0;


// =====================================================
//                       SETUP
// =====================================================

void setup() {

  Serial.begin(9600);

  gsm.begin(9600);

  dht.begin();

  pinMode(RELAY_PIN, OUTPUT);

  // Pump OFF at startup
  digitalWrite(RELAY_PIN, LOW);

  delay(1000);

  Serial.println();
  Serial.println("=================================");
  Serial.println("          PlantPulse AI");
  Serial.println("     SMART PLANT MONITORING");
  Serial.println("=================================");

  // ---------------------------------------------------
  // GSM INITIALIZATION
  // ---------------------------------------------------

  gsm.println("AT");
  delay(500);

  gsm.println("ATE0");
  delay(500);

  gsm.println("AT+CMGF=1");
  delay(500);

  // Clear old GSM responses
  while (gsm.available()) {
    gsm.read();
  }

  Serial.println("GSM Initialized");
  Serial.println("System Started");
  Serial.println("---------------------------------");
}


// =====================================================
//                        LOOP
// =====================================================

void loop() {

  // ---------------------------------------------------
  // Read sensors every 1 second
  // ---------------------------------------------------

  if (millis() - lastSensorRead >= SENSOR_INTERVAL) {

    lastSensorRead = millis();

    readSensors();

    printStatus();

    controlPump();

    sendPythonData();
  }


  // ---------------------------------------------------
  // SAFETY CHECK
  // ---------------------------------------------------

  if (pumpState == true) {

    if (millis() - pumpStartTime >= MAX_PUMP_RUNTIME) {

      stopPumpSafety();
    }
  }
}


// =====================================================
//                  READ ALL SENSORS
// =====================================================

void readSensors() {

  // ===================================================
  // SOIL MOISTURE
  // ===================================================

  soilRaw = analogRead(SOIL_PIN);

  currentMoisture = map(
    soilRaw,
    SOIL_DRY_RAW,
    SOIL_WET_RAW,
    0,
    100
  );

  currentMoisture = constrain(
    currentMoisture,
    0,
    100
  );


  // ===================================================
  // HUMIDITY
  // ===================================================

  float newHumidity = dht.readHumidity();

  if (!isnan(newHumidity)) {

    humidity = newHumidity;
  }


  // ===================================================
  // LIGHT
  // ===================================================

  lightRaw = analogRead(LDR_PIN);

  /*
     More light = lower analog value
     Less light = higher analog value
  */

  lightLevel = map(
    lightRaw,
    1023,
    0,
    0,
    100
  );

  lightLevel = constrain(
    lightLevel,
    0,
    100
  );
}


// =====================================================
//                 WATER PUMP CONTROL
// =====================================================

void controlPump() {

  // ===================================================
  // START WATERING
  // ===================================================

  if (
    currentMoisture < WATER_START &&
    pumpState == false
  ) {

    // Save moisture BEFORE watering
    previousMoisture = currentMoisture;

    Serial.println();
    Serial.println("=================================");
    Serial.println("SOIL IS DRY!");
    Serial.println("STARTING WATERING");
    Serial.println("=================================");


    // -------------------------------------------------
    // SEND START SMS FIRST
    // -------------------------------------------------

    bool smsOK = sendSMS(
      "PlantPulse AI\n"
      "Watering started.\n"
      "Soil: " + String(previousMoisture) + "%"
    );

    if (smsOK) {
      Serial.println("Start SMS: OK");
    } else {
      Serial.println("Start SMS: FAILED");
    }


    // -------------------------------------------------
    // TURN PUMP ON
    // -------------------------------------------------

    digitalWrite(RELAY_PIN, HIGH);

    pumpState = true;

    pumpStartTime = millis();
  }


  // ===================================================
  // STOP WATERING
  // ===================================================

  if (
    currentMoisture >= WATER_STOP &&
    pumpState == true
  ) {

    stopPump();
  }
}


// =====================================================
//                  NORMAL PUMP STOP
// =====================================================

void stopPump() {

  // ---------------------------------------------------
  // TURN PUMP OFF FIRST
  // ---------------------------------------------------

  digitalWrite(RELAY_PIN, LOW);

  pumpState = false;


  Serial.println();
  Serial.println("=================================");
  Serial.println("TARGET MOISTURE REACHED");
  Serial.println("WATERING COMPLETED");
  Serial.println("PUMP OFF");
  Serial.println("=================================");


  // ---------------------------------------------------
  // SEND SHORT COMPLETION SMS
  // ---------------------------------------------------

  bool smsOK = sendSMS(
    "PlantPulse AI\n"
    "Watering completed.\n"
    "Soil: " + String(currentMoisture) + "%"
  );

  if (smsOK) {
    Serial.println("Completion SMS: OK");
  } else {
    Serial.println("Completion SMS: FAILED");
  }
}


// =====================================================
//             SAFETY PUMP STOP
// =====================================================

void stopPumpSafety() {

  digitalWrite(RELAY_PIN, LOW);

  pumpState = false;


  Serial.println();
  Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
  Serial.println("PUMP SAFETY TIMEOUT");
  Serial.println("PUMP FORCED OFF");
  Serial.println("CHECK SOIL SENSOR / WATER SUPPLY");
  Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");


  // ---------------------------------------------------
  // SHORT SAFETY SMS
  // ---------------------------------------------------

  bool smsOK = sendSMS(
    "PlantPulse AI\n"
    "SAFETY ALERT!\n"
    "Pump stopped after 30 min.\n"
    "Check sensor/water."
  );

  if (smsOK) {
    Serial.println("Safety SMS: OK");
  } else {
    Serial.println("Safety SMS: FAILED");
  }
}


// =====================================================
//                 SEND DATA TO PYTHON
// =====================================================

void sendPythonData() {

  /*
     Python receives:

     DATA,soil,humidity,light,pump

     Example:

     DATA,42,65.5,78,0
  */

  Serial.print("DATA,");

  Serial.print(currentMoisture);
  Serial.print(",");

  Serial.print(humidity, 1);
  Serial.print(",");

  Serial.print(lightLevel);
  Serial.print(",");

  Serial.println(
    pumpState ? 1 : 0
  );
}


// =====================================================
//                  SERIAL MONITOR
// =====================================================

void printStatus() {

  Serial.println();
  Serial.println("---------------------------------");

  Serial.print("Soil Raw      : ");
  Serial.println(soilRaw);

  Serial.print("Soil Moisture : ");
  Serial.print(currentMoisture);
  Serial.println("%");

  Serial.print("Humidity      : ");
  Serial.print(humidity, 1);
  Serial.println("%");

  Serial.print("Light Raw     : ");
  Serial.println(lightRaw);

  Serial.print("Light Level   : ");
  Serial.print(lightLevel);
  Serial.println("%");

  Serial.print("Pump          : ");

  if (pumpState) {

    Serial.println("ON");

  } else {

    Serial.println("OFF");
  }
}


// =====================================================
//                     SEND SMS
// =====================================================

bool sendSMS(String message) {

  Serial.println();
  Serial.println("GSM: Sending SMS...");


  // ---------------------------------------------------
  // CLEAR OLD GSM DATA
  // ---------------------------------------------------

  while (gsm.available()) {
    gsm.read();
  }


  // ---------------------------------------------------
  // SMS TEXT MODE
  // ---------------------------------------------------

  gsm.println("AT+CMGF=1");

  if (!waitForResponse("OK", 3000)) {

    Serial.println("GSM ERROR: CMGF");
    return false;
  }


  // ---------------------------------------------------
  // SET RECIPIENT
  // ---------------------------------------------------

  gsm.print("AT+CMGS=\"");
  gsm.print(phoneNumber);
  gsm.println("\"");


  // ---------------------------------------------------
  // WAIT FOR > PROMPT
  // ---------------------------------------------------

  if (!waitForPrompt(5000)) {

    Serial.println("GSM ERROR: No > prompt");
    return false;
  }


  // ---------------------------------------------------
  // SEND MESSAGE
  // ---------------------------------------------------

  gsm.print(message);


  // ---------------------------------------------------
  // CTRL + Z
  // ---------------------------------------------------

  gsm.write(26);


  // ---------------------------------------------------
  // WAIT FOR SMS RESULT
  // ---------------------------------------------------

  if (waitForSMSResult(10000)) {

    Serial.println("GSM: SMS confirmed.");
    return true;

  } else {

    Serial.println("GSM ERROR: SMS not confirmed.");
    return false;
  }
}


// =====================================================
//                WAIT FOR GSM PROMPT
// =====================================================

bool waitForPrompt(unsigned long timeout) {

  unsigned long startTime = millis();

  while (millis() - startTime < timeout) {

    if (gsm.available()) {

      char c = gsm.read();

      Serial.write(c);

      if (c == '>') {

        return true;
      }
    }
  }

  return false;
}


// =====================================================
//               WAIT FOR GSM RESPONSE
// =====================================================

bool waitForResponse(
  const char* expected,
  unsigned long timeout
) {

  unsigned long startTime = millis();

  byte match = 0;
  byte expectedLength = strlen(expected);

  while (millis() - startTime < timeout) {

    if (gsm.available()) {

      char c = gsm.read();

      Serial.write(c);

      if (c == expected[match]) {

        match++;

        if (match >= expectedLength) {

          return true;
        }

      } else {

        match = 0;
      }
    }
  }

  return false;
}


// =====================================================
//              WAIT FOR SMS RESULT
// =====================================================

bool waitForSMSResult(unsigned long timeout) {

  unsigned long startTime = millis();

  char buffer[32];
  byte index = 0;

  while (millis() - startTime < timeout) {

    while (gsm.available()) {

      char c = gsm.read();

      Serial.write(c);

      // Store response characters
      if (index < sizeof(buffer) - 1) {

        buffer[index++] = c;
        buffer[index] = '\0';
      }

      // Successful SMS
      if (strstr(buffer, "+CMGS:") != NULL) {

        // Wait briefly for final OK
        unsigned long okStart = millis();

        while (millis() - okStart < 5000) {

          if (gsm.available()) {

            char r = gsm.read();

            Serial.write(r);

            if (r == 'O') {

              if (gsm.available()) {

                char k = gsm.read();

                Serial.write(k);

                if (k == 'K') {

                  return true;
                }
              }
            }
          }
        }

        return true;
      }

      // GSM error
      if (strstr(buffer, "ERROR") != NULL) {

        return false;
      }

      if (strstr(buffer, "+CMS ERROR") != NULL) {

        return false;
      }
    }
  }

  return false;
}
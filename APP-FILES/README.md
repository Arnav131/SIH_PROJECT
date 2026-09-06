# APP-FILES — UrbanSenseAI Android client

`app-debug.apk` is versionCode 5 / 1.1.1, installed and exercised on a Realme RMX1851 running
Android 11. Paths below show where each source file belongs inside the `android-app/` project.

## Contents

| File | Goes to | What it does |
|---|---|---|
| `app-debug.apk` | — | Installable build |
| `src/Core.kt` | `app/src/main/java/com/urbansenseai/` | Sensors, GPS, WebSocket client, offline queue, app state |
| `src/ShockDetector.kt` | `app/src/main/java/com/urbansenseai/` | Decides when a jolt deserves a photograph |
| `src/MainActivity.kt` | `app/src/main/java/com/urbansenseai/` | UI, permissions, CameraX preview and capture |
| `src/CollectionService.kt` | `app/src/main/java/com/urbansenseai/` | Foreground service that survives the screen turning off |
| `res/AndroidManifest.xml` | `app/src/main/` | Permissions and foreground-service type |
| `res/styles.xml` | `app/src/main/res/values/` | App theme |
| `res/strings.xml` | `app/src/main/res/values/` | App name |
| `test/ShockDetectorTest.kt` | `app/src/test/java/com/urbansenseai/` | 16 tests over the shock detector |
| `test/EndpointTest.kt` | `app/src/test/java/com/urbansenseai/` | 12 tests over the receiver-URL rules |
| `gradle/app-build.gradle` | `app/build.gradle` | Module config and dependencies |
| `gradle/root-build.gradle` | `build.gradle` | AGP and Kotlin plugin versions |
| `gradle/settings.gradle` | `settings.gradle` | Repositories and module list |
| `gradle/gradle.properties` | `gradle.properties` | JVM args, AndroidX flag |

## Install and build

```
adb install -r app-debug.apk
```

```
cd android-app
gradlew :app:testDebugUnitTest :app:assembleDebug
```

## Auto-capture on shock

A sharp, repeated jolt — a pothole, a kerb, a hard brake — photographs itself and tags the
frame with the current fix. Controls sit under **AUTO-CAPTURE ON SHOCK**: an on/off button, a
sensitivity button cycling Low → Medium → High, and a live readout of the current score, the
events fired this session, and the last one.

**The screen must stay on and the app in front.** CameraX is bound to the Activity lifecycle,
so a locked phone cannot take a picture. Accelerometer, gyroscope, GPS and transmission all
keep running in the background through the foreground service — only the photography stops.

The camera photographs whatever it is pointed at, so mount the phone facing the road before a
real run; face-up on a seat it will fill the gallery with the ceiling.

### How the trigger decides

Gravity is a constant ~9.8 on whichever axis points down, so raw magnitude says nothing about
movement. A slow running average tracks gravity and is subtracted, leaving only the dynamic
part of the acceleration.

That dynamic magnitude is judged against **its own recent baseline** rather than a fixed
number. A phone rattling in a car mount settles into a high baseline and stops tripping, while
a genuinely sharp jolt still stands out against it. The baseline only learns from calm samples
— folding a jolt back in would raise the bar exactly as a rough stretch begins.

A single sample over the line is almost always sensor noise, so a burst is required: at least
3 hits inside 1.2 seconds. After firing, a 4-second cooldown stops one rough patch producing a
hundred photographs. Six or more hits in the window is labelled `shock_sustained`.

| Sensitivity | Fires above baseline | Accelerometer floor | Gyroscope floor |
|---|---|---|---|
| Low | 6× | 4.5 m/s² | 2.2 rad/s |
| Medium (default) | 4× | 2.5 m/s² | 1.3 rad/s |
| High | 3× | 1.4 m/s² | 0.8 rad/s |

A reading clears the floor when **either** the accelerometer or the gyroscope is over its own
limit. That makes a sharp twist enough on its own, which is right in a vehicle but means
casually picking the phone up off a desk also fires — a hand rotates it well past 1.3 rad/s.
Drop to Low for desk testing, or change `overFloor` in `ShockDetector.kt` to require the
accelerometer floor and let the gyroscope only raise the score.

`ShockDetectorTest` drives the detector with sampled motion at the real 20 Hz rate: a still
phone never fires, walking never fires, an isolated spike is ignored, a burst fires exactly
once, continuous shaking respects the cooldown, a rattling mount settles into the baseline,
and a hard pothole still breaks through that rattle.

### Capture size

Captures are capped to 1280×960 by a `ResolutionSelector`. The sensor's default is 12 MP,
which put 400–650 KB on the phone's uplink per event and made the browser decode a
multi-megapixel JPEG per gallery card. Road damage is legible at 1280×960 and the frames land
around 150 KB.

### What a shock capture sends

```json
{"type":"camera","filename":"shock_1788550000000.jpg","width":1280,"height":960,
 "trigger":"shock","accel_magnitude":13.4,"gyro_magnitude":1.82,"shock_score":7.6,
 "latitude":30.7655999,"longitude":76.5748425,"gps_accuracy":8.5,"gps_provider":"gps"}
```

Then the JPEG as a binary frame. A manual CAPTURE sends `"trigger":"manual"` and still carries
the fix — a photograph is never stored without the place it was taken.

## Receiver URL rules

- On a LAN use `http://<laptop-ip>:8000` — **not** `https://`. A plain server cannot answer a
  TLS handshake and the failure is silent from the phone's side.
- `https://` belongs to tunnel URLs (`https://….trycloudflare.com`) only.
- A bare `192.168.1.5:8000` is accepted and assumed to be `http`.
- The laptop's IP comes from DHCP and changes. The server prints the current one at startup.

## Bugs these files fix

The project did not compile before — there was never an APK.

- Three compile errors: a missing `toHttpUrl` import, a `Request.url()` overload ambiguity,
  and a local `fun` after a `;` on one line.
- The socket URL was given to OkHttp as `ws://…`, which `HttpUrl` rejects outright, so the
  connection never opened. It is dialled as `http(s)://` now; only the UI label says `ws`.
- The theme was a platform Material theme under an `AppCompatActivity`, which threw on launch.
- The service declared a `camera` foreground type without the matching permission.
- `collectLatest` at 40 Hz cancelled every render before the main thread ran it, so X/Y/Z never
  repainted while the status line did.
- GPS subscribed to `GPS_PROVIDER` only with no last-known-fix seed, so it sat at 0.000000
  indoors. The network provider is subscribed alongside it now.

/*
Flow Kerjanya begini gan (per frame/per tensor/per prediksi)
1. convert tensor hasil prediksi MoveNet ke array dulu pakai getKeypoints()
2. array nanti dimasukkan ke model klasifikasi pose
3. update arm angle atau sudut-sudut lainnya pakai fungsi armAngle(inputnya array tadi) dkk.
4. update jumlah repetisi pakai fungsi-fungsi cek repetisi (sesuai jenis workout hasil klasifikasi pose)
5. cek form user pakai fungsi checkSquat atau checkPushup sesuai klasifikasi pose
*/

import com.bangkit.bangkitcapstone.pose.tflite.Pose_Classifier
import kotlin.math.atan2
import kotlin.math.PI

// Init dependencies
val KEYPOINT_DICT = mapOf(
    "nose" to Pair(0, 1),
    "left_eye" to Pair(2, 3),
    "right_eye" to Pair(4, 5),
    "left_ear" to Pair(6, 7),
    "right_ear" to Pair(8, 9),
    "left_shoulder" to Pair(10, 11),
    "right_shoulder" to Pair(12, 13),
    "left_elbow" to Pair(14, 15),
    "right_elbow" to Pair(16, 17),
    "left_wrist" to Pair(18, 19),
    "right_wrist" to Pair(20, 21),
    "left_hip" to Pair(22, 23),
    "right_hip" to Pair(24, 25),
    "left_knee" to Pair(26, 27),
    "right_knee" to Pair(28, 29),
    "left_ankle" to Pair(30, 31),
    "right_ankle" to Pair(32, 33)
)

poseModel = "pose_model.tflite"
private lateinit var classifier: Pose_Classifier

var isUpSquat = false
var isDownSquat = false
var squat_reps = 0

var isUpPushup = false
var isDownPushup = false
var pushup_reps = 0

// override fun onCreate(savedInstanceState: Bundle?) {
//     super.onCreate(savedInstanceState)
//     initClassifier()
// }

private fun initClassifier() {
    classifier = Pose_Classifier(assets, poseModel)
}

// Extract keypoints coordinate in x and y of 17 points
val keypoints_array = getKeypoints(# OUTPUT MOVENET)

// Predict pose from the extracted points array
val result = classifier.predictPose(keypoints_array)

// Calculate keypoints angles for squat
if (result <= 1.0 && result > 0.95) {
    val knee_angle = kneeAngle(keypoints_array)
    
    squat_reps = upSquat(knee_angle, squat_reps)
    downSquat(knee_angle)
}

// Calculate keypoints angles for pushup
else if (result <= 5.0 && result >= 0.0) {
    val elbow_angle = armAngle(keypoints_array)
    val back_angle = backAngle(keypoints_array)
    
    // Check user in up position or down position and add reps if user is going up and down
    pushup_reps = upPushup(elbow_angle, pushup_reps)
    downPushup(elbow_angle)

    // Check user's pushup form (0 if back position is bad, 1 if back position is good)
    pushup_form = checkPushup(keypoints_array)
}

// Fungsi buat ekstrak koordinat keypoint dari output tensor hasil prediksi MoveNet
fun getKeypoints(keypointsArray: FloatArray): List<Float> {
    val keyList = mutableListOf<Float>()
    for (i in 0 until 50 step 3) {
        val x = keypointsArray.get(i + 1)
        val y = keypointsArray.get(i)
        keyList.add(x)
        keyList.add(y)
    }
    return keyList
}

// ------------------------------ Fungsi Cek Sudut Pushup

// Fungsi buat ngitung sudut sikut saat pushup
fun armAngle(keypoints: List<Float>): Float {
    // Input berupa list yang sudah dikonvert dengan fungsi getKeypoints()
    // Output berupa sudut dari sikut
    val leftWristIndex = KEYPOINT_DICT["left_wrist"]
    val leftShoulderIndex = KEYPOINT_DICT["left_shoulder"]
    val leftElbowIndex = KEYPOINT_DICT["left_elbow"]

    val at1 = atan2(
        keypoints[leftWristIndex!!.second] - keypoints[leftElbowIndex!!.second],
        keypoints[leftWristIndex.first] - keypoints[leftElbowIndex.first]
    )
    val at2 = atan2(
        keypoints[leftShoulderIndex!!.second] - keypoints[leftElbowIndex.second],
        keypoints[leftShoulderIndex.first] - keypoints[leftElbowIndex.first]
    )
    val angleTot = (at1 - at2) * (180 / PI.toFloat())
    return angleTot
}

// Fungsi buat ngitung kelurusan punggung (ngitung sudut pinggang) saat pushup
fun backAngle(keypoints: List<Float>): Float {
    // Input berupa list yang sudah dikonvert dengan fungsi getKeypoints()
    // Output berupa sudut dari pinggang
    val leftHipIndex = KEYPOINT_DICT["left_hip"]
    val leftShoulderIndex = KEYPOINT_DICT["left_shoulder"]
    val leftKneeIndex = KEYPOINT_DICT["left_knee"]

    val at1 = atan2(
        keypoints[leftKneeIndex!!.second] - keypoints[leftHipIndex!!.second],
        keypoints[leftKneeIndex.first] - keypoints[leftHipIndex.first]
    )
    val at2 = atan2(
        keypoints[leftShoulderIndex!!.second] - keypoints[leftHipIndex.second],
        keypoints[leftShoulderIndex.first] - keypoints[leftHipIndex.first]
    )
    val angleTot = (at1 - at2) * (180 / PI.toFloat())
    return angleTot
}

// ------------------------------ Fungsi Cek Sudut Squat

// Fungsi ini buat ngitung sudut lutut (squat)
fun kneeAngle(keypoints: List<Float>): Float {
    // Input berupa list yang sudah dikonvert dengan fungsi getKeypoints()
    // Output berupa sudut dari tekukan lutut
    val leftHipIndex = KEYPOINT_DICT["left_hip"]
    val leftAnkleIndex = KEYPOINT_DICT["left_ankle"]
    val leftKneeIndex = KEYPOINT_DICT["left_knee"]

    val at1 = atan2(
        keypoints[leftAnkleIndex!!.second] - keypoints[leftKneeIndex!!.second],
        keypoints[leftAnkleIndex.first] - keypoints[leftKneeIndex.first]
    )
    val at2 = atan2(
        keypoints[leftHipIndex!!.second] - keypoints[leftKneeIndex.second],
        keypoints[leftHipIndex.first] - keypoints[leftKneeIndex.first]
    )
    val angleTot = (at1 - at2) * (180 / PI.toFloat())
    return angleTot
}

// Fungsi ini buat ngukur sudut pinggang saat squat
fun hipAngle(keypoints: List<Float>): Float {
    // Input berupa list yang sudah dikonvert dengan fungsi getKeypoints()
    // Output berupa sudut dari pinggang
    val leftHipIndex = KEYPOINT_DICT["left_hip"]
    val leftShoulderIndex = KEYPOINT_DICT["left_shoulder"]
    val leftKneeIndex = KEYPOINT_DICT["left_knee"]

    val at1 = atan2(
        keypoints[leftKneeIndex!!.second] - keypoints[leftHipIndex!!.second],
        keypoints[leftKneeIndex.first] - keypoints[leftHipIndex.first]
    )
    val at2 = atan2(
        keypoints[leftShoulderIndex!!.second] - keypoints[leftHipIndex.second],
        keypoints[leftShoulderIndex.first] - keypoints[leftHipIndex.first]
    )
    val angleTot = (at1 - at2) * (180 / PI.toFloat())
    return angleTot
}

// ------------------------------ Fungsi Cek Repetisi Pushup

// Fungsi untuk mengecek apakah user pada posisi naik pushup
fun upPushup(elbowAngle: Float, repsVar: Int): Int {
    // Input berupa hasil hitungan dari armAngle()
    var reps = repsVar
    if (!((Math.abs(elbowAngle) > 170) && (Math.abs(elbowAngle) < 200))) {
        return reps
    }

    if (isDownPushup) {
        println("up")
        reps++
    }

    isUpPushup = true
    isDownPushup = false

    return reps
}

// Fungsi untuk mengecek apakah user sedang pada posisi bawah pushup
fun downPushup(elbowAngle: Float) {
    // Input berupa hasil hitungan dari fungsi armAngle()
    if (!((Math.abs(elbowAngle) > 50) && (Math.abs(elbowAngle) < 90))) {
        // exit function
        return
    }

    if (isUpPushup) {
        println("down")
    }

    isUpPushup = false
    isDownPushup = true
}

fun checkPushup(backAngle: Float): Int {
    // Input berupa hasil hitungan dari backAngle()
    // measure appropriate back angle and returns warning code (1 OK, 0 BAD)
    if (!(Math.abs(backAngle) > 160)) {
        return 0
    }
    return 1
}

// ------------------------------ Fungsi Cek Repetisi Squat

fun checkSquat(hipAngle: Float): Int {
    // Input dari hasil fungsi hipAngle()
    // measure appropriate hip angle and returns warning code (1 OK, 0 BAD)
    if (!((Math.abs(hipAngle) > 60) && (Math.abs(hipAngle) < 80))) {
        return 0
    }
    return 1
}

// Fungsi untuk ngecek apakah user di posisi squat berdiri
fun upSquat(kneeAngle: Float, repsVar: Int): Int {
    var reps = repsVar
    if (!((Math.abs(kneeAngle) > 170) && (Math.abs(kneeAngle) < 195))) {
        return reps
    }

    if (isDownSquat) {
        println("up")
        reps++
    }

    isUpSquat = true
    isDownSquat = false

    return reps
}

// Fungsi untuk ngecek posisi user sedang jongkok saat squat
fun downSquat(kneeAngle: Float) {
    if (!((Math.abs(kneeAngle) > 255) && (Math.abs(kneeAngle) < 280))) {
        // exit function
        return
    }

    if (isUpSquat) {
        println("down")
    }

    isUpSquat = false
    isDownSquat = true
}
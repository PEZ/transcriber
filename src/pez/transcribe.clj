(ns pez.transcribe
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [com.phronemophobic.whisper :as whisper]))

(comment
  ;; First init the model - using load-model instead of init
  (def get-text (whisper/record-and-transcribe "models/ggml-base.en.bin"))

  ;; Then use the initialized model for recording
  (def get-text (whisper/record-and-transcribe "models/ggml-model-q5_0.bin"))
  (def get-text (whisper/record-and-transcribe "models/ggml-model.bin"))

  ;; Get the transcription when ready
  (def transcription (get-text))

  (def transcription (whisper/transcribe-wav
                      "models/ggml-model-q5_0.bin"
                      #_"models/ggml-model.bin"
                      "test-data/chatgpt-rovarspraksfunderinar.wav"))

  :rcf)
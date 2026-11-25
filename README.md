# 人物跟蹤系統（仿富萱科技 AiUnion 安全監控行為辨識）

固定監視鏡頭 → YOLO11n 人物偵測 → ByteTrack 多人跟蹤 → 電子圍籬入侵／禁區逗留／劃線進出計數。

## 執行

```powershell
.\.venv\Scripts\python.exe src\main.py --source data\vtest.avi --config configs\vtest.json --out outputs\vtest --model models\yolo11n.pt
```

加 `--show` 可即時顯示畫面（Esc 離開）；`--tracker botsort.yaml` 可換追蹤器。

## 輸出（`outputs/<名稱>/`）

| 檔案 | 內容 |
|---|---|
| `annotated.mp4` | 標註框、ID、軌跡、禁區、計數線、即時告警（禁區內持續紅框＋頂端紅色 `ALERT` 橫幅） |
| `events.csv` | 每筆事件：frame、秒數、類型（intrusion / loitering / line_in / line_out）、track_id、目標 |
| `alerts/` | 每次觸發 `intrusion` / `loitering` 當下自動留存的告警截圖（`fXXXX_idXX.jpg`） |
| `summary.json` | 幀數、ID 數、平均 FPS、各線進出數、各類事件數 |

## 設定（`configs/*.json`）

- `zones`：禁區多邊形（像素座標）與 `loiter_sec` 逗留門檻
- `lines`：計數線 `a`→`b`，人從線的左側走到右側算 `in`
- `exit_frames`：連續在禁區外幾幀才算離開（防止邊界抖動重複告警）
- `line_margin_px`：離計數線超過幾像素才算確定在某一側（防止沿線走時來回重複計數）
- `line_confirm_frames`：換到另一側要連續幾次才算跨線（防止 ID 接錯一幀造成一進一出）
- `lost_frames`：跟蹤遺失幾幀後清除狀態

判斷位置用的是框的**腳底中心點**，不是框中心，這樣比較貼近人實際站的位置。

## 追蹤器對照（2026-11-25，vtest.avi，yolo11n）

```powershell
.\.venv\Scripts\python.exe src\compare_trackers.py outputs\cmp_bytetrack outputs\cmp_botsort outputs\cmp_botsort_reid
```

| 設定 | ID 數 | 疑似 ID 切換（handoff） | 平均軌跡秒數 | FPS |
|---|---|---|---|---|
| ByteTrack（預設） | 34 | 15 | 13.3 | 26.7 |
| BoT-SORT（不開 ReID，GMC 關閉） | 34 | 15 | 13.3 | 25.9 |
| BoT-SORT + ReID（auto） | 38 | 17 | 11.9 | 23.3 |
| ByteTrack，track_buffer 60 | 34 | 14 | 13.3 | 26.0 |
| ByteTrack，conf 0.1 | 34 | 14 | 13.4 | 25.3 |
| ByteTrack，conf 0.1、imgsz 960 | 45 | 21 | 10.6 | 24.8 |

結論：ByteTrack 維持預設。ultralytics 的 BoT-SORT 只在 IoU ≥ 0.5（proximity_thresh）時才參考外觀特徵，
人被遮擋後再出現時已經移動，IoU 過不了門檻，ReID 根本派不上用場；
另外 auto 模式用的是 YOLO 骨幹特徵，在 50–80px 的小人框上鑑別力不足，反而讓 ID 數增加。
15 次切換中有 9 次間隔 ≤ 1 秒、全部都在 track_buffer 內，所以問題出在關聯門檻，不是緩衝太短。

> 指標是無標註資料下的代理指標，不是 MOTA/IDF1。

## 遺失軌跡找回（`--recovery`）

追蹤器給出新 ID 時，`src/recovery.py` 會在剛消失（3 秒內）的人裡面找最像的接回去：

1. **位置閘門**：用消失前的速度預測現在該在哪，半徑隨消失時間放寬（40px + 8px/幀）
2. **框高**：新舊框高度比 ≤ 1.6
3. **外觀**：上衣、褲子各一個 HSV 直方圖；上衣相似度 < 0.47 直接排除，平均 ≥ 0.55 才接
4. **收回**：若追蹤器之後自己把原軌跡接回來（同一幀出現兩個相同 ID），較早的留下、後接的拆回自己的 ID
5. 與別人重疊時不更新外觀，避免被別人的衣服污染

| 設定 | ID 數 | 疑似 ID 切換 | 重複 ID 幀 | 平均軌跡秒數 |
|---|---|---|---|---|
| ByteTrack | 34 | 15 | 0 | 13.3 |
| BoT-SORT + ReID | 38 | 17 | 0 | 11.9 |
| **ByteTrack + 找回** | **28** | **10** | 0 | **16.1** |

每次找回都會輸出 `merges.csv` 與 `merges_gallery.jpg`（左：消失前、右：新出現），方便人工檢查有沒有接錯人。
注意：handoff 指標和找回用的是類似的「時間＋距離」條件，數字下降有一部分是方法本身決定的，接得對不對要看圖庫。

## 劃線計數稽核（`src/line_audit.py`）

不重跑 YOLO，直接拿 `tracks.csv` 重播計數邏輯，可掃參數並輸出每次跨線的前後截圖：

```powershell
.\.venv\Scripts\python.exe src\line_audit.py --run outputs\vtest --video data\vtest.avi --config configs\vtest.json --margins 0 10 --confirms 1 2 3 --gallery-margin 10
```

| 設定 | ByteTrack | ByteTrack + 找回 |
|---|---|---|
| 舊版（純線段相交） | 12/8 | 14/9 |
| 遲滯 10px、確認 2 次（目前預設） | 12/8 | 13/8 |

- 舊版多算的：有人沿著線走、在線附近來回（一進一出），以及 ID 接錯一幀又收回造成的一進一出
- 確認次數 3 太嚴：剛跨線就被遮擋的人會漏算
- 兩者仍差的 1 次：有人在燈柱後面被擋住時跨線，ByteTrack 斷成兩個 ID 所以沒算到，找回把它接起來

**場景建議**：計數線目前畫在燈柱上，剛好是最容易遮擋、ID 最容易斷的地方。把線往左或右平移 80px，
兩種追蹤方式算出來完全一致，代表遠離遮擋物的線比較可靠。

## 危險區域入侵稽核（`src/zone_audit.py`）

不重跑 YOLO，直接用 `tracks.csv` 重播禁區事件並產出每次入侵的前／當下／後截圖（`intrusion_audit.jpg`）：

```powershell
.\.venv\Scripts\python.exe src\zone_audit.py --run outputs\vtest --video data\vtest.avi --config configs\vtest.json
```

`vtest.avi`（草地 `NO-ENTRY`）逐筆核對結果（共 6 筆入侵＋1 筆逗留）：
- **即時觸發**：真正走進草地的人（`ID 53`、`ID 59`、`ID 71`）都在腳底跨入禁區當下（1 幀內，約 0.1 秒）立即觸發紅框、頂端橫幅與截圖存證；`ID 59` 停留滿 5.0 秒（`53.1s`→`58.1s`）準確觸發逗留告警。
- **稽核修掉的邊界 Bug**：原本多邊形底邊設在 `y=576`（剛好壓在畫面底緣），導致 `ID 59` 走到畫面底部時被射線法判為離開、再觸發第二次入侵；改為 `y=580` 後，重複入侵消除（7→6 筆），連續在區內 217 幀無中斷。
- **已知邊界條件（誠實限制）**：
  1. `ID 53` 與 `ID 59` 在草地內近距離重疊遮擋後，`ID 53` 僅露上半身（框高減半），超出找回門檻 `max_height_ratio=1.6`，因此切出新 `ID 61` 並多報 1 次入侵（人對人嚴重遮擋是輕量色彩直方圖找回的極限）。
  2. `ID 1`（`0.0s`）、`ID 47`（`60.6s`）為腳底貼著草地邊緣擦邊 3–6 幀；若實務上不希望擦邊告警，可將多邊形內縮或加上進入確認幀數。
  3. 遺失軌跡找回適合短時間遮擋（≤ 3 秒）且衣著有區別的場景；若遇**穿同色衣服交錯**或**光線劇烈變化**，需升級為專用 ReID 模型（如 OSNet）。

## 測試

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

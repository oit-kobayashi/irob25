# self localization server

## API メモ

- PUT /robots/{id}
  body: {pose: [x, y, theta], sigma: [s^2_x, s^2_y, s^2_th]}
  resp: なし
  ロボットの**新規作成**。分散共分散行列は(面倒なので)対角成分のみ
  
- GET /robots/{id}
  resp: robot {id: id, {pose: [x, y, th], sigma: [[]]}}
  ロボットの pose と**位置の分散共分散行列を取得**。
  
- POST /robots/{id}/pred_update
  body: {left: float, right: float, sll: float, srr: float}
  resp: なし
  **予測更新** left, right は左右輪の移動量, sll, srrはそれらの誤差分散。

- POST /robots/{id}/perc_update
  body: 
  resp: なし
  **計測更新**

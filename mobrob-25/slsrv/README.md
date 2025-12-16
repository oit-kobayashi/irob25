# self localization server

## API メモ

### ロボットの新規作成

- PUT /robots/{id}
  body: {pose: [x, y, theta], sigma: [s^2_x, s^2_y, s^2_th]}
  resp: なし
  ロボットの新規作成。分散共分散行列は(面倒なので)対角成分のみ
  
- GET /robots/{id}
  resp: robot {id: id, {pose: [x, y, th], sigma: [[]]}}
  ロボットの pose と位置の分散共分散行列を取得。
  
- PUT /robots/{id}

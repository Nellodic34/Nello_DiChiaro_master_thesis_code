# sim/ — harness di simulazione (gitignorato, non arriva mai ai droni)

Fa girare **i nodi di questo repo** dentro la simulazione FlightForge, senza
toccare nessun file tracciato: la config di deploy resta quella che vola.

## Uso

Una volta sola, nell'albero sim:

    cd ~/git/training/mrs_uav_flightforge_simulator/tmux/one_drone
    patch launch_sim.sh < ~/vision_stack_deploy/sim/launch_sim.patch

Poi, sempre:

    ~/vision_stack_deploy/sim/run_sim.sh

## Perche' serve un harness e non basta un env

I nodi risolvono la propria config da soli, come
`Path(__file__).parent.parent / 'config' / 'runtime_stack.yaml'`. Quindi **quali
nodi esegui decide quale config viene letta**, e non esiste variabile d'ambiente
che possa puntarli a uno YAML diverso. Con i nodi di questo repo la config e'
sempre quella di deploy: nomi droni reali, topic RealSense, pesi reali.

Quello che si puo' sovrascrivere da fuori:

| cosa | come |
|---|---|
| nomi droni, ruoli, sim time, knob KF | variabili d'ambiente, gia' previste dai launcher |
| `model_path`, `device`, `publish_camera_static_tf`, `camera_frame_override` | parametri ROS dichiarati, passati via `"$@"` |
| topic immagine e camera_info | **solo remap ROS**: non sono parametri, li legge dallo YAML |

Il terzo caso e' il motivo per cui questi shim esistono.

## Chi comanda cosa

Due YAML, due competenze distinte, e non si sovrappongono:

- **lo YAML della sim** (`one_drone/config/runtime_stack.yaml`) decide il *mondo*:
  quanti droni nascono (`drone_count: 3`), come si chiamano (uav1/uav2/uav3), il
  simulatore, e se `launch_sim.sh` apre il pannello evaluator.
- **lo YAML di questo repo** (`config/runtime_stack.yaml`) decide il *comportamento*:
  tutti i knob di visione, triangolazione e KF. E' quello che vola, invariato.

I nomi dei droni li leggono gli shim direttamente da `uav_roles` dello YAML sim, quindi
se cambi quel mondo qui non si tocca niente.

Quello che l'harness **forza di proposito**, e che quindi non arriva dallo YAML di
deploy: `use_sim_time` (true, contro il false che serve ai droni veri),
`drone_run_mode` (`decentralized`), `model_path` (`weights_sim`), `device` (`cuda`),
`publish_camera_static_tf` (false). Tutto il resto - KF, triangolazione, visione -
e' esattamente quello che vola.

## Le quattro trappole

1. **Tipo del messaggio.** `observer_node` decide `CompressedImage` o `Image` dal
   suffisso `/compressed` del nome *configurato*, non di quello remappato. Dopo il
   remap continua quindi a volere compressed, mentre FlightForge pubblica raw:
   `ensure_compressed` avvia un republish solo se il topic compresso non c'e' gia'.
2. **TF della camera.** In deploy il nodo pubblica da se' la catena fcu->camera con
   gli offset di montaggio del drone vero. In sim quella catena esiste gia', quindi
   gli shim passano `publish_camera_static_tf:=false` e lasciano che il frame arrivi
   dal `CameraInfo` del simulatore.
3. **Pesi.** `weights8` e' addestrato su riprese reali; sull'immagine sintetica puo'
   non agganciare nulla. Gli shim puntano di default a `weights_sim` dell'albero sim.
4. **Non solo le immagini vanno remappate.** `live_monitor.py` non dichiara *nessun*
   parametro ROS e costruisce i marker degli observer come
   `/{uav_roles observer}/{odom_topic_suffix}` dallo yaml di deploy: in sim aspettava
   `/uav11` e `/uav8` e i triangoli non comparivano, mentre target e stima si vedevano
   perche' arrivano da `/target/...`, indipendente dai nomi. Ogni topic costruito a
   partire da un nome drone va remappato, non solo quelli della camera.

## Cosa non e' stato verificato

Gli script sono coerenti con quello che i nodi leggono davvero (nomi dei remap
confrontati con lo YAML), ma **non sono mai stati eseguiti contro una sim viva**:
serve GPU, FlightForge e un terminale grafico. Il primo avvio va guardato a mano.
Il punto piu' probabile di rottura e' il numero 2: se non escono raggi, guardare
`ros2 run tf2_tools view_frames` e il frame dichiarato nel `CameraInfo` di uav1.

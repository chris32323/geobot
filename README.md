# GeoGuess-Reel-Bot

Postet jeden Tag automatisch ein Instagram-Reel: Zwischentitel („WHERE ARE WE?“ / „STILL IN?“), 360°-Schwenk durch ein **echtes Mapillary-Panorama**, **echte Weltkarte** mit Timer, Zoom zur Auflösung mit 1.000-km-Kreis und Abschlussfrage. Dazu spricht eine **männliche KI-Stimme** auf Englisch. Der Bot läuft komplett auf **GitHub Actions**, du brauchst keinen eigenen Server.

```
GitHub Actions (täglich)
  ├─ Ort aus data/locations.json wählen (keine Wiederholung in den letzten 60 Posts)
  ├─ echtes 360°-Panorama in der Nähe über die Mapillary-API holen
  ├─ KI-Stimme erzeugen (Piper, männlich, kostenlos – oder OpenAI „onyx“)
  ├─ Video rendern (Natural-Earth-Weltkarte, ffmpeg)
  ├─ Video als GitHub Release hochladen  → öffentliche URL
  ├─ Reel über die Instagram-API posten
  └─ data/history.json committen (merkt sich die Orte und hält den Zeitplan aktiv)
```

## Einrichtung (einmalig, ca. 30–45 Minuten)

### 1. GitHub-Repo
1. Neues **öffentliches** Repository anlegen und den Inhalt dieses Ordners hochladen. Öffentlich muss es sein, damit Instagram das Video aus dem Release abrufen kann. Deine Zugangsdaten liegen als Secrets trotzdem sicher.
2. Unter **Settings → Actions → General → Workflow permissions** „Read and write permissions“ auswählen.

### 2. Mapillary-Token (kostenlos)
1. Auf mapillary.com einloggen → **Dashboard → Developers → Register application**.
2. Den **Client Token** kopieren (beginnt mit `MLY|`).
3. Im Repo unter **Settings → Secrets and variables → Actions → New repository secret** als `MAPILLARY_TOKEN` speichern.

### 3. Instagram
1. Das Instagram-Konto auf **Business-Konto** umstellen (in der App: Einstellungen → Kontotyp und Tools).
2. Auf developers.facebook.com eine App anlegen und den Use Case **„Instagram API“ (mit Instagram-Login)** hinzufügen.
3. Dein Instagram-Konto in der App als Tester bzw. mit einer Rolle hinzufügen und die Einladung in Instagram annehmen.
4. Einen Token mit den Berechtigungen `instagram_business_basic` und `instagram_business_content_publish` erzeugen. Er muss **langlebig** sein (60 Tage).
5. Zwei Secrets anlegen:
   - `IG_ACCESS_TOKEN`: der Token
   - `IG_USER_ID`: deine Instagram-User-ID. Du bekommst sie über `https://graph.instagram.com/me?fields=user_id,username&access_token=DEIN_TOKEN`.

### 4. Token automatisch erneuern
Instagram-Tokens laufen nach 60 Tagen ab. Der Workflow `refresh-token.yml` erneuert sie jeden Montag. Dafür braucht er die Erlaubnis, das Secret zu überschreiben:
1. GitHub → Settings → Developer settings → **Fine-grained personal access token**, nur für dieses Repo, Berechtigung **Secrets: Read and write**.
2. Diesen Token als Secret `GH_PAT` speichern.

### 5. Testlauf
**Actions → Daily GeoGuess Reel → Run workflow** mit `dry_run` = an. Das Video wird gerendert und unter **Releases** abgelegt, aber **nicht** gepostet. Schau es dir an. Wenn alles passt, startest du einen Lauf mit `dry_run` = aus. Danach postet der Bot jeden Tag selbstständig.

## Einstellungen (optional)
Unter **Settings → Secrets and variables → Actions → Variables**:

| Variable | Standard | Bedeutung |
|---|---|---|
| `ROUNDS_PER_REEL` | `2` | Runden pro Reel (bei 3 kommt „LAST CHANCE!“) |
| `TTS_ENGINE` | `piper` | `piper` (kostenlos) oder `openai` (klingt natürlicher, kostet wenige Cent; dafür das Secret `OPENAI_API_KEY` anlegen) |
| `PIPER_VOICE` | `en_US-ryan-high` | männliche Piper-Stimme, Alternative z. B. `en_GB-alan-medium` |

- **Uhrzeit:** in `.github/workflows/daily-reel.yml` die `cron`-Zeile ändern (die Zeit ist in UTC).
- **Orte:** in `data/locations.json` ergänzen oder entfernen. `difficulty` kann `easy`, `medium` oder `hard` sein, Runde 1 bekommt leichtere Orte. `hint` ist der Hinweis, den die Stimme beim Schwenk sagt. Orte, an denen Mapillary kein gutes 360°-Bild hat, merkt sich der Bot unter `no_coverage` und lässt sie künftig aus.
- **Radius:** `RADIUS_KM` (Standard 1000).

## Lokal testen
```bash
pip install -r requirements.txt && bash scripts/fetch_assets.sh
python -m bot.main render --out out --demo --fake-voice   # offline, ohne Tokens, mit Platzhalter-Panoramen aus demo/
MAPILLARY_TOKEN=MLY|... python -m bot.main render --out out   # echte Bilder + echte Stimme
```

## Wichtig
- **Lizenz:** Mapillary-Bilder stehen unter CC BY-SA 4.0. Der Bot blendet die Quelle im Video ein und schreibt sie in die Beschreibung, bitte nicht entfernen. Die Kartendaten von Natural Earth sind gemeinfrei. Die Schrift Poppins steht unter der SIL Open Font License.
- **GitHub-Zeitpläne** können sich bei hoher Last etwas verspäten. Der tägliche Commit von `history.json` verhindert, dass GitHub den Zeitplan nach 60 Tagen ohne Aktivität abschaltet.
- **Wenn ein Lauf fehlschlägt** (kein Panorama gefunden, Token abgelaufen, …), schickt GitHub dir eine E-Mail. Es wird dann nichts Halbfertiges gepostet.
- Instagram erlaubt höchstens 100 API-Posts pro 24 Stunden. Ein Post am Tag ist weit darunter.

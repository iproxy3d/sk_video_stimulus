# Публикация пакета в PyPI

Пакет публикуется через **PyPI Trusted Publishing**. GitHub Actions получает одноразовый короткоживущий ключ через OIDC, поэтому постоянный API-токен не хранится ни в репозитории, ни в GitHub Secrets.

## 1. Отозвать ранее созданный токен

Если API-токен PyPI когда-либо передавался в чат, сообщение или открытый файл, его необходимо считать скомпрометированным:

1. откройте `https://pypi.org/manage/account/token/`;
2. удалите старый токен;
3. не создавайте новый токен для GitHub Actions — workflow использует Trusted Publishing.

## 2. Зарегистрировать ожидаемого издателя в PyPI

Откройте `https://pypi.org/manage/account/publishing/`, найдите форму добавления ожидаемого издателя GitHub Actions и укажите:

| Поле PyPI | Значение |
|---|---|
| Имя проекта PyPI | `sk-video-stimulus` |
| Владелец репозитория | `iproxy3d` |
| Имя репозитория | `sk_video_stimulus` |
| Имя workflow | `publish-pypi.yml` |
| Окружение GitHub | `pypi` |

Имя проекта должно полностью совпадать со значением `project.name` в `pyproject.toml`. Ожидаемый издатель не резервирует имя до первой успешной публикации.

## 3. Настроить окружение GitHub

В репозитории откройте **Settings → Environments → New environment** и создайте окружение `pypi`.

Рекомендуется включить обязательное ручное подтверждение публикации через **Required reviewers**. Код и тесты будут собраны автоматически, но загрузка в PyPI начнётся только после подтверждения.

Не добавляйте `PYPI_API_TOKEN`: workflow `.github/workflows/publish-pypi.yml` не использует постоянные секреты.

## 4. Локально проверить первую версию

Из корня репозитория выполните:

```bash
python -m pip install --upgrade pip build twine
python -m pip install -e ".[all,dev]"
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```

Должны появиться два файла:

```text
dist/sk_video_stimulus-0.3.0-py3-none-any.whl
dist/sk_video_stimulus-0.3.0.tar.gz
```

## 5. Опубликовать версию `0.2.0`

1. Убедитесь, что изменения находятся в ветке `main`, а GitHub Actions `tests` завершился успешно.
2. Откройте страницу **Releases** репозитория.
3. Нажмите **Draft a new release**.
4. Создайте тег `v0.3.0` из ветки `main`.
5. Заголовок выпуска укажите `sk-video-stimulus 0.3.0`.
6. Нажмите **Publish release**.
7. Откройте запущенный workflow **Публикация в PyPI** и подтвердите окружение `pypi`, если включена ручная проверка.

Первая успешная публикация автоматически создаст проект `sk-video-stimulus` в PyPI и преобразует ожидаемого издателя в обычного доверенного издателя.

## 6. Проверить установку из PyPI

После завершения workflow создайте чистое виртуальное окружение и выполните:

```bash
python -m venv .venv-test
```

Windows PowerShell:

```powershell
.venv-test\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "sk-video-stimulus[all]"
python -c "import sk_video_stimulus as s; print(s.__version__)"
sk-video-stimulus --help
sk-video-stimulus-plot --help
sk-video-stimulus-demo --help
sk-video-stimulus-video --help
sk-video-stimulus-webcam --help
```

Linux и macOS:

```bash
source .venv-test/bin/activate
python -m pip install --upgrade pip
python -m pip install "sk-video-stimulus[all]"
python -c "import sk_video_stimulus as s; print(s.__version__)"
sk-video-stimulus --help
sk-video-stimulus-plot --help
sk-video-stimulus-demo --help
sk-video-stimulus-video --help
sk-video-stimulus-webcam --help
```

Ожидаемая версия: `0.3.0`.

## Следующие выпуски

Номер уже опубликованной версии нельзя использовать повторно. Для следующего выпуска:

1. измените `project.version` в `pyproject.toml`, например на `0.3.1`;
2. отправьте изменения в `main` и дождитесь успешных тестов;
3. создайте новый GitHub Release с тегом `v0.3.1`.

`sk_video_stimulus.__version__` читается из метаданных установленного пакета, поэтому отдельно менять версию в `src/sk_video_stimulus/__init__.py` не требуется.

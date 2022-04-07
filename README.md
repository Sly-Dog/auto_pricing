# Алгоритм приноса изменений

1. `git switch <своя ветка>`
2. `git add <свои файлы>`
3. `git commit -m"что сделал"`
4. `git switch main`
5. `git pull`
6. `git switch <своя ветка>`
7. `git rebase`
8. `git merge main`  просмотр, все ли работает у тебя
9. `git switch main`
10. `git merge <своя ветка>`

Можно вводить сразу несколько команд
```
git switch main
git switch second
```


# Минимальный сет для командной работы
## Начало работы
1. Скопировать репозиторий на компьютер. Появится папка там, где был открыт git bash
```
git clone https://github.com/Sly-Dog/auto_pricing
```
Ждем полной загрузки до тех пор, пока не появится возможность вводить новую команду
После загрузки - закрываем

2. Открываем папку auto_pricing в ней жмякаем git bash - Готово
## Наиболее частые команды
Создать ветку. Ветки используем для любой фичи. Для любого отдельного ноутбука. Они нужны для разделения работы.
```
git checkout -b <name of the branch>
```
---

Узнать, какие есть ветки
```
git branch
```
---
Перейти в ветку для того, чтобы работать в ней.
```
git checkout <name of the branch>
```
---
Узнать статус изменений
```
git status
```
---
Выбрать изменения в комит
```
git add <filename>
```
---
Выбрать все изменения в комит
```
git add .
```
---

Отменяет добавление в комит
```
git reset
```
---
Удалить незафиксированные изменения
[git clean](https://www.atlassian.com/ru/git/tutorials/undoing-changes/git-clean)
---
Сделать комит
```
git commit -m"Информативное название комита"
```
---
Ура! Можно отразить изменения на сервере, чтобы остальные могли узнать, что произошло
```
git push
```
---
Запросить изменения для того, чтобы влить в основую ветку
```
git pull
```

## Работа с ветками
Нарисовать граф изменений со всеми ветками. Там будут отображаться хеши -
длинные коды
```
git log --graph --all
```

хеши - по сути тоже названия веток, они отражают предыдущие изменения. 
`git checkout hash` - откратиться на выбранное изменение

---

Если ты уверен и ответственнен, то ты можешь сливать ветки сам. Cледующий
код вливает указанную ветку в ту, на которой ты сейчас находишься. Так можно
посмотреть самому, правильно ли все в логике ноутбуков.
```
git merge <branch>
```
Есть метод rebase, он обновляет основную ветку для комитов твоей ветки, вот как это выглядт
![](https://joprblob.azureedge.net/site/blog/50fa5f40-93ac-475e-895d-8724cc761d19/rebase.gif)
---
[Анимации методов](https://bool.dev/blog/detail/vizualizatsiya-poleznykh-git-komand)

## Чтобы разобраться, если есть пробелы
- [Как установить](https://youtu.be/GsG5roSGha0)
- [Как (и для чего) использовать систему контроля версий git](https://youtu.be/wvqiGJu3YmQ)
- [Разбираемся как ходить по комиттам git: checkout](https://youtu.be/jXKdGIV7O3w)
- [бранчевание/ветвление в git](https://youtu.be/GYVXVu7qgCE)
- [Git - Pull request на практике / GitHub](https://youtu.be/G_HKJJLozUc)

[Курс полноценный](https://youtu.be/xzEMA7rzN3Y)

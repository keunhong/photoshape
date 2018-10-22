
for f in $(find $1); do
  python -m terial.commands.exemplar.register "$f"
done


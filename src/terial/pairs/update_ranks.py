from terial.database import session_scope


def main():
    with session_scope() as sess:
        result = sess.execute("""
    update exemplar_shape_pair p set rank = s.rank from (
    select id, rank() over (partition by shape_id order by distance asc)
    from exemplar_shape_pair
        ) s
    where p.id = s.id;
    """)
    print(str(result))


if __name__ == '__main__':
    main()

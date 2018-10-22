--
-- PostgreSQL database dump
--

-- Dumped from database version 10.3 (Debian 10.3-1.pgdg90+1)
-- Dumped by pg_dump version 10.5

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: envmaps; Type: TABLE; Schema: public; Owner: photoshape_user
--

CREATE TABLE public.envmaps (
    id integer NOT NULL,
    name character varying,
    azimuth double precision DEFAULT 0,
    enabled boolean DEFAULT false,
    source character varying,
    split_set character varying
);


ALTER TABLE public.envmaps OWNER TO photoshape_user;

--
-- Name: envmaps_id_seq; Type: SEQUENCE; Schema: public; Owner: photoshape_user
--

CREATE SEQUENCE public.envmaps_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.envmaps_id_seq OWNER TO photoshape_user;

--
-- Name: envmaps_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: photoshape_user
--

ALTER SEQUENCE public.envmaps_id_seq OWNED BY public.envmaps.id;


--
-- Name: envmaps id; Type: DEFAULT; Schema: public; Owner: photoshape_user
--

ALTER TABLE ONLY public.envmaps ALTER COLUMN id SET DEFAULT nextval('public.envmaps_id_seq'::regclass);


--
-- Data for Name: envmaps; Type: TABLE DATA; Schema: public; Owner: photoshape_user
--

INSERT INTO public.envmaps VALUES (6, 'studio021', 3.14158999999999988, true, 'deviantart', 'train');
INSERT INTO public.envmaps VALUES (2, 'studio017', 0, true, 'deviantart', 'validation');
INSERT INTO public.envmaps VALUES (18, 'Studio_Umbrellas_B', 0, true, 'adobe-stock', 'train');
INSERT INTO public.envmaps VALUES (3, 'studio018', 3.14158999999999988, true, 'deviantart', 'train');
INSERT INTO public.envmaps VALUES (19, 'Studio_Windows_Dark', 0, true, 'adobe-stock', 'train');
INSERT INTO public.envmaps VALUES (1, 'studio016', 1.57079499999999994, true, 'deviantart', 'train');
INSERT INTO public.envmaps VALUES (16, 'Studio_Windows_Bright', 0, true, 'adobe-stock', 'train');
INSERT INTO public.envmaps VALUES (15, 'Studio_Softboxes_And_Lamp', 0, true, 'adobe-stock', 'train');
INSERT INTO public.envmaps VALUES (10, 'studio025', 3.14158999999999988, true, 'deviantart', 'train');
INSERT INTO public.envmaps VALUES (11, 'studio026', 3.14158999999999988, true, 'deviantart', 'train');
INSERT INTO public.envmaps VALUES (4, 'studio019', 3.14158999999999988, true, 'deviantart', 'validation');
INSERT INTO public.envmaps VALUES (17, 'Studio_Windows_Medium', 0, true, 'adobe-stock', 'validation');
INSERT INTO public.envmaps VALUES (13, 'studio028', 3.14158999999999988, true, 'deviantart', 'train');
INSERT INTO public.envmaps VALUES (12, 'studio027', 0, true, 'deviantart', 'validation');
INSERT INTO public.envmaps VALUES (31, 'HdrStudioProductBlueAndYellow001', 0, false, 'poliigon', NULL);
INSERT INTO public.envmaps VALUES (20, 'uffizi', 0, false, 'debevec', NULL);
INSERT INTO public.envmaps VALUES (21, 'building', 0, false, 'debevec', NULL);
INSERT INTO public.envmaps VALUES (22, 'ennis', 0, false, 'debevec', NULL);
INSERT INTO public.envmaps VALUES (23, 'kitchen', 0, false, 'debevec', NULL);
INSERT INTO public.envmaps VALUES (24, 'rnl', 0, false, 'debevec', NULL);
INSERT INTO public.envmaps VALUES (25, 'galileo', 0, false, 'debevec', NULL);
INSERT INTO public.envmaps VALUES (26, 'campus', 0, false, 'debevec', NULL);
INSERT INTO public.envmaps VALUES (27, 'stpeters', 0, false, 'debevec', NULL);
INSERT INTO public.envmaps VALUES (28, 'grace', 0, false, 'debevec', NULL);
INSERT INTO public.envmaps VALUES (29, 'fila', 0, false, 'debevec', NULL);
INSERT INTO public.envmaps VALUES (30, 'kitchen', 0, false, 'debevec', NULL);
INSERT INTO public.envmaps VALUES (8, 'studio023', 0, false, 'deviantart', NULL);
INSERT INTO public.envmaps VALUES (9, 'studio024', 0, false, 'deviantart', NULL);
INSERT INTO public.envmaps VALUES (7, 'studio022', 0, false, 'deviantart', NULL);
INSERT INTO public.envmaps VALUES (5, 'studio020', 0, false, 'deviantart', NULL);
INSERT INTO public.envmaps VALUES (14, 'studio029', 3.14158999999999988, false, 'deviantart', NULL);


--
-- Name: envmaps_id_seq; Type: SEQUENCE SET; Schema: public; Owner: photoshape_user
--

SELECT pg_catalog.setval('public.envmaps_id_seq', 31, true);


--
-- Name: envmaps envmaps_pkey; Type: CONSTRAINT; Schema: public; Owner: photoshape_user
--

ALTER TABLE ONLY public.envmaps
    ADD CONSTRAINT envmaps_pkey PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

